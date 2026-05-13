## Hessian Definition
```math
\begin{aligned}
H \equiv \nabla^2 f(x) + \sum_{i=1}^{n_c} y_{c,i} \nabla^2 c_i(x) + \sum_{i=1}^{n_d} y_{d,i} \nabla^2 d_i(x)
\end{aligned}
```
Here $`H`$ is the Hessian, $`f(x)`$ is the objective function, $`y_{c,i}`$ are the equality constraints and $`y_{d,i}`$ are the inequality constraints.
See the explicit derivatives [here](/docs/opflow/pbpol.md).
## Input

ExaGO OPFLOW reads .m file, thus the input file for unit test is in this format.
A 5-bus system **CICJ-unittestx1.m** will be used as a basis for this test. In addition, an artifical solution vector will be also generated as an input for the test.

### Hessian sparsity structure:

The Hessian has the following sparsity structure for our test:
![Hessian_sparsity.png](/tests/unit/opflow/Hessian/Hessian_sparsity.png)
