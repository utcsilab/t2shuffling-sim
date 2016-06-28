# T2 Shuffling Simulation Tools
Tools for simulating spin-echo trains and computing temporal subspaces.

## typical use case
```bash
./main.py --T2vals ../data/kneesim.mat --genFSE --rvc --numT2 256 --angles ../data/flipangles.txt.180 --ETL 78 --TE 5555e-6 --e2s 2 --model simple_svd --save_basis /Users/jtamir/Desktop/ --set_basis_name bas.e2s2.opetl80.flip50
```

## Other things you can do
. Make a realistic analytical contrast phantom (in image space)
. Compute a contrast synthesis matrix for the subspace

# Contributing
Contributing is encouraged. Please do not directly commit to `master`. Instead, create and push a branch, and open a
pull request.
