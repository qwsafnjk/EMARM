# EMARM
## Project Structure

The project entrypoints are:
- test.py: is the file for testing the trained EMARM.
- train.py: is the entrypoint for training EMARM.

All the other files contain helper functions and utilities.

The FJSSP instances are divided into two folders:
- dataset5k: contains the instances used for training models.
- benchmarks: contains the test and validation instances. 


## Dataset and benchmark instances

All the instances used for training and testing follow the same structure.
Here is a small example:

```
3 2           
0 9 13 17 1 6 9 12
1 5 8 9 0 7 8 10
1 3 5 6 0 3 4 5
10
```

The first line gives the number of jobs and machines in the instance. 
In this example, the instance has 3 jobs and 2 machines. 

Then, follow information about the jobs in the instance. Each job is 
given as a sequence of pairs (machine index, fuzzy processing time). 
For example, the first job starts executing on machine 0 for (9,12,13) time units,
and afterward it goes on machine 1 for (6,9,12) time units. The second and third jobs 
follow the same structure.

After the instance, a rough upper bound is given for scaling the instance.


