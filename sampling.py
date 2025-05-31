import torch
import torch.nn.functional as F


class FJSSPStates:
    """
    FJSSP state for parallel executions.

    Args:
        device: Where to create tensors.
    """
    # Number of features in the internal state
    size = 11

    def __init__(self, device: str = 'cpu', eps: float = 1e-5):
        self.num_j = None       # Number of jobs
        self.num_m = None       # Number of machines
        self.machines = None    # Machine assigment of each operation
        self.costs1 = None       # Cost of each operation
        self.costs2 = None
        self.costs3 = None
        self._factor = None     # Max cost
        self._eps = eps
        self._q = torch.tensor([0.25, 0.5, 0.75], device=device)
        
        self.dev = device       # Tensor device
        self._K_idx = None     # Sampling index for accessing info
        self.K = None          # Sampling number
        
        self.j_ct1 = None        # Completion time of jobs in the partial sol
        self.j_ct2 = None
        self.j_ct3 = None
        self.j_idx = None       # Index of active operation in jobs
        self.j_st = None        
        
        self.m_ct1 = None        # Completion time of machines in the partial sol
        self.m_ct2 = None
        self.m_ct3 = None

    def init_state(self, ins: dict, K: int = 1):
        """
        Initialize the state of the FJSSP.

        Args:
            ins: FJSSP instance.
            K: Sampling number.
        Return:
            - The parallel states.
            - The mask of active operations for each state.
        """
        self.num_j, self.num_m = ins['j'], ins['m']
        self.machines = ins['machines'].view(-1).to(self.dev)
        self._factor = ins['costs3'].max()
        self.costs1 = ins['costs1'].view(-1).to(self.dev) / self._factor
        self.costs2 = ins['costs2'].view(-1).to(self.dev) / self._factor
        self.costs3 = ins['costs3'].view(-1).to(self.dev) / self._factor
        self.K = K
        self._K_idx = torch.arange(K, device=self.dev)
        
        self.j_st = torch.arange(0, self.num_j * self.num_m, self.num_m,
                                 device=self.dev)
        self.j_idx = torch.zeros((K, self.num_j), dtype=torch.int32,
                                 device=self.dev)
        self.j_ct1 = torch.zeros((K, self.num_j), dtype=torch.float32,
                                device=self.dev)
        self.j_ct2 = torch.zeros((K, self.num_j), dtype=torch.float32,
                                device=self.dev)
        self.j_ct3 = torch.zeros((K, self.num_j), dtype=torch.float32,
                                device=self.dev)
        
        self.m_ct1 = torch.zeros((K, self.num_m), dtype=torch.float32,
                                device=self.dev)
        self.m_ct2 = torch.zeros((K, self.num_m), dtype=torch.float32,
                                device=self.dev)
        self.m_ct3 = torch.zeros((K, self.num_m), dtype=torch.float32,
                                device=self.dev)

        # Create the initial state and mask
        states = torch.zeros((K, self.num_j, self.size), dtype=torch.float32,  
                             device=self.dev)

        return states, self.mask.to(torch.float32)

    @property
    def mask(self):
        """
        Boolean mask that points out the uncompleted jobs.

        Return:
            Tensor with shape (K, num jobs).
        """
        return self.j_idx < self.num_m

    @property
    def ops(self):
        """
        The index of active/ready operations for each job.
        Note that for the completed job the active operation is the one with
        index 0.

        Return:
            Tensor with shape (bs, num jobs).
        """
        return self.j_st + (self.j_idx % self.num_m)

    @property
    def makespan(self):
        """
        Compute defuzzy makespan of solutions.
        """
        m_ct = (self.m_ct1 + 2 * self.m_ct2 + self.m_ct3) / 4.0 + (self.m_ct3 - self.m_ct1) * 0.4
        return m_ct.max(-1)[0] * self._factor
    
    @property
    def fuzzymakespan(self):
        """
        Compute fuzzy makespan of solutions.
        """
        m_ct = (self.m_ct1 + 2 * self.m_ct2 + self.m_ct3) / 4.0 + (self.m_ct3 - self.m_ct1) * 0.4
        temp = m_ct.max(-1)[0]
        idx = temp.min(-1)[1]
        idx2 = m_ct[idx].max(-1)[1]
        return [self.m_ct1[idx][idx2] * self._factor, self.m_ct2[idx][idx2] * self._factor, self.m_ct3[idx][idx2] * self._factor]
    
    def __schedule__(self, jobs: torch.Tensor):
        """ Schedule the selected jobs and update fuzzy completion times. """
        _idx = self._K_idx           # sampling index
        _ops = self.ops[_idx, jobs]   # Active operations of selected jobs
        macs = self.machines[_ops]    # Machines of active operations

        # Update fuzzy completion times
        condition = ((-0.15 * self.m_ct1[_idx, macs] + 0.5 * self.m_ct2[_idx, macs] + 0.65 * self.m_ct3[_idx, macs]) >
                     (-0.15 * self.j_ct1[_idx, jobs] + 0.5 * self.j_ct2[_idx, jobs] +  0.65 * self.j_ct3[_idx, jobs]))
        ct1 = self.costs1[_ops] + self.m_ct1[_idx, macs] * condition + self.j_ct1[_idx, jobs] * (~condition)
        ct2 = self.costs2[_ops] + self.m_ct2[_idx, macs] * condition + self.j_ct2[_idx, jobs] * (~condition)
        ct3 = self.costs3[_ops] + self.m_ct3[_idx, macs] * condition + self.j_ct3[_idx, jobs] * (~condition)

        self.m_ct1[_idx, macs] = ct1
        self.m_ct2[_idx, macs] = ct2
        self.m_ct3[_idx, macs] = ct3
        self.j_ct1[_idx, jobs] = ct1
        self.j_ct2[_idx, jobs] = ct2
        self.j_ct3[_idx, jobs] = ct3

        # Activate the following operation on job, if any
        self.j_idx[_idx, jobs] += 1

    def update(self, jobs: torch.Tensor):
        """
        Update the state at training

        Args:
            jobs: Index of the job.
                Shape (K).
        """
        # Schedule the selected operations
        self.__schedule__(jobs)

        job_mac = self.machines[self.ops]  # Machines of active ops
        m_ct = (self.m_ct1 + 2 * self.m_ct2 + self.m_ct3) / 4.0 + (self.m_ct3 - self.m_ct1) * 0.4
        mac_ct = m_ct.gather(1, job_mac)  
        j_ct = (self.j_ct1 + 2 * self.j_ct2 + self.j_ct3) / 4.0 + (self.j_ct3 - self.j_ct1) * 0.4
        curr_ms = j_ct.max(-1, keepdim=True)[0] + self._eps
        
        n_states = -torch.ones((self.K, self.num_j, self.size),
                               device=self.dev)
        n_states[..., 0] = j_ct - mac_ct
        # Distance of each job from quantiles computed among all jobs
        q_j = torch.quantile(j_ct, self._q, -1).T
        n_states[..., 1:4] = j_ct.unsqueeze(-1) - q_j.unsqueeze(1)
        n_states[..., 4] = j_ct - j_ct.mean(-1, keepdim=True)
        n_states[..., 5] = j_ct / curr_ms
        # Distance of each job from quantiles computed among all machines
        q_m = torch.quantile(m_ct, self._q, -1).T
        n_states[..., 6:9] = mac_ct.unsqueeze(-1) - q_m.unsqueeze(1)
        n_states[..., 9] = mac_ct - m_ct.mean(-1, keepdim=True)
        n_states[..., 10] = mac_ct / curr_ms
      
        return n_states, self.mask.to(torch.float32)

    def __call__(self, jobs: torch.Tensor, states: torch.Tensor):
        """
        Update the state at inference.

        Args:
            jobs: Index of the job.
                Shape (K).
        """
        # Schedule the selected operations
        self.__schedule__(jobs)

        job_mac = self.machines[self.ops]  # Machines of active ops
        m_ct = (self.m_ct1 + 2 * self.m_ct2 + self.m_ct3) / 4.0 + (self.m_ct3 - self.m_ct1) * 0.4
        mac_ct = m_ct.gather(1, job_mac)
        j_ct = (self.j_ct1 + 2 * self.j_ct2 + self.j_ct3) / 4.0 + (self.j_ct3 - self.j_ct1) * 0.4
        curr_ms = j_ct.max(-1, keepdim=True)[0] + self._eps
        
        states[..., 0] = j_ct - mac_ct
        # Distance of each job from quantiles computed among all jobs
        q_j = torch.quantile(j_ct, self._q, -1).T
        states[..., 1:4] = j_ct.unsqueeze(-1) - q_j.unsqueeze(1)
        states[..., 4] = j_ct - j_ct.mean(-1, keepdim=True)
        states[..., 5] = j_ct / curr_ms
        # Distance of each job from quantiles computed among all machines
        q_m = torch.quantile(m_ct, self._q, -1).T
        states[..., 6:9] = mac_ct.unsqueeze(-1) - q_m.unsqueeze(1)
        states[..., 9] = mac_ct - m_ct.mean(-1, keepdim=True)
        states[..., 10] = mac_ct / curr_ms

        return self.mask.to(torch.float32)


@torch.no_grad()
def sampling(ins: dict,
             encoder: torch.nn.Module,
             decoder: torch.nn.Module,
             K: int = 32,
             device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Encoder.
        decoder: Decoder
        K: sampling number.
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    machines = ins['machines'].view(-1)
    encoder.eval()
    decoder.eval()

    # Reserve space for the solution
    sols = -torch.ones((K, num_m, num_j), dtype=torch.long, device=device)
    _idx = torch.arange(K, device=device)
    m_idx = torch.zeros((K, num_m), dtype=torch.long, device=device)

    jsp = FJSSPStates(device)
    state, mask = jsp.init_state(ins, K)

    # Encoding step
    embed = encoder(ins['x'], edge_index=ins['edge_index'])

    # Decoding steps, (in the last step, there is only one job to schedule)
    for i in range(num_j * num_m - 1):
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.multinomial(1, replacement=False).squeeze(1)

        # Add the selected operations to the solution matrices
        s_ops = ops[_idx, jobs]
        m = machines[s_ops]
        s_idx = m_idx[_idx, m]
        sols[_idx, m, s_idx] = s_ops
        m_idx[_idx, m] += 1
        # Update the context of the solutions
        mask = jsp(jobs, state)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)
    return sols, jsp.makespan

@torch.no_grad()
def sampling_fuzzy(ins: dict,
             encoder: torch.nn.Module,
             decoder: torch.nn.Module,
             K: int = 32,
             device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Encoder.
        decoder: Decoder
        K: sampling number.
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    machines = ins['machines'].view(-1)
    encoder.eval()
    decoder.eval()

    # Reserve space for the solution
    sols = -torch.ones((K, num_m, num_j), dtype=torch.long, device=device)
    _idx = torch.arange(K, device=device)
    m_idx = torch.zeros((K, num_m), dtype=torch.long, device=device)
    
    jsp = FJSSPStates(device)
    state, mask = jsp.init_state(ins, K)

    # Encoding step
    embed = encoder(ins['x'], edge_index=ins['edge_index'])

    # Decoding steps, (in the last step, there is only one job to schedule)
    for i in range(num_j * num_m - 1):
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.multinomial(1, replacement=False).squeeze(1)

        # Add the selected operations to the solution matrices
        s_ops = ops[_idx, jobs]
        m = machines[s_ops]
        s_idx = m_idx[_idx, m]
        sols[_idx, m, s_idx] = s_ops
        m_idx[_idx, m] += 1
        # Update the context of the solutions
        mask = jsp(jobs, state)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)
    return sols, jsp.fuzzymakespan

def sample_training(ins: dict,
                    encoder: torch.nn.Module,
                    decoder: torch.nn.Module,
                    K: int = 32,
                    device: str = 'cpu'):
    """
    Sample multiple trajectories while training.

    Args:
        ins: The instance to solve.
        encoder: Encoder.
        decoder: Decoder
        K: sampling number.
        device: Either cpu or cuda.
    """
    encoder.train()
    decoder.train()
    num_j, num_m = ins['j'], ins['m']
    # We don't need to learn from the last step, everything is masked but a job
    num_ops = num_j * num_m - 1

    # Reserve space for the solution
    tarjs = -torch.ones((K, num_ops), dtype=torch.long, device=device)  
    ptrs = -torch.ones((K, num_ops, num_j), dtype=torch.float32, 
                       device=device)
    
    jsp = FJSSPStates(device)
    state, mask = jsp.init_state(ins, K)

    # Encoding step
    embed = encoder(ins['x'].to(device),
                    edge_index=ins['edge_index'].to(device))

    # Decoding steps
    for i in range(num_ops):
        # Generate logits and make the completed jobs
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()   
                                                      
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.multinomial(1, replacement=False).squeeze(1)

        tarjs[:, i] = jobs 
        ptrs[:, i] = logits
        
        state, mask = jsp.update(jobs)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)
    return tarjs, ptrs, jsp.makespan
