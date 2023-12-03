from pydrake.solvers import MathematicalProgram, Solve
import numpy as np

def traj_prog(x0, xf, N, n_circle, circles, radius):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(N, 2, "x")
    u = prog.NewContinuousVariables(N - 1, 2, "u")

    prog.AddBoundingBoxConstraint(np.zeros((N, 2)), np.ones((N, 2)), x)
    prog.AddBoundingBoxConstraint(-np.ones((N - 1, 2)) * 0.05, np.ones((N - 1, 2)) * 0.05, u)

    prog.AddConstraint(x[0, 0] == x0[0])
    prog.AddConstraint(x[0, 1] == x0[1])
    prog.AddConstraint(x[-1, 0] == xf[0])
    prog.AddConstraint(x[-1, 1] == xf[1])

    for i in range(N):
        for j in range(n_circle):
            prog.AddConstraint(np.linalg.norm(x[i] - circles[j]) >= radius[j])

    for i in range(N - 1):
        prog.AddConstraint(x[i + 1, 0] == x[i, 0] + u[i, 0])
        prog.AddConstraint(x[i + 1, 1] == x[i, 1] + u[i, 1])

    R = np.eye(2)
    Rt = np.eye(2)
    for i in range(N - 1):
        prog.AddCost(u[i].T @ R @ u[i])
        if i < N - 2:
            prog.AddCost((u[i + 1] - u[i]).T @ Rt @ (u[i + 1] - u[i]))

    return prog, x


def traj_obs(n_circle, circles, radius):
    def solve(prog, x, trajs, costs):
        result = Solve(prog)
        if result.is_success():
            x_result = result.GetSolution(x)
            cost_result = result.get_optimal_cost()
            if len(costs) == 0 or np.linalg.norm((np.stack(trajs) - x_result[np.newaxis, :, :]), axis=2).sum(axis=1).min() > 5e-1:
                trajs.append(x_result)
                costs.append(cost_result)
    N = 100
    x0 = np.array([0, 0])
    xf = np.array([1, 1])
    prog, x = traj_prog(x0, xf, N, n_circle, circles, radius)
    trajs1 = []
    costs1 = []
    for i in np.linspace(-10, 10, 11):
        init_x = np.linspace(x0[0], xf[0], N)
        prog.SetInitialGuess(x[:, 0], init_x)
        if i < 0:
            prog.SetInitialGuess(x[:, 1], init_x ** (-1 / i))
        else:
            prog.SetInitialGuess(x[:, 1], init_x ** i)
        solve(prog, x, trajs1, costs1)
    prog.SetInitialGuess(x[:N // 2], np.linspace(x0, np.array([1, 0]), N // 2))
    prog.SetInitialGuess(x[N // 2:], np.linspace(np.array([1, 0]), xf, N - N // 2))
    solve(prog, x, trajs1, costs1)
    prog.SetInitialGuess(x[:N // 2], np.linspace(x0, np.array([0, 1]), N // 2))
    prog.SetInitialGuess(x[N // 2:], np.linspace(np.array([0, 1]), xf, N - N // 2))
    solve(prog, x, trajs1, costs1)

    N = 100
    x0 = np.array([0, 1])
    xf = np.array([1, 0])
    prog, x = traj_prog(x0, xf, N, n_circle, circles, radius)
    trajs2 = []
    costs2 = []
    for i in np.linspace(-10, 10, 11):
        init_x = np.linspace(x0[0], xf[0], N)
        prog.SetInitialGuess(x[:, 0], init_x)
        if i < 0:
            prog.SetInitialGuess(x[:, 1], 1 - init_x ** (-1 / i))
        else:
            prog.SetInitialGuess(x[:, 1], 1 - init_x ** i)
        solve(prog, x, trajs2, costs2)
    prog.SetInitialGuess(x[:N // 2], np.linspace(x0, np.array([0, 0]), N // 2))
    prog.SetInitialGuess(x[N // 2:], np.linspace(np.array([0, 0]), xf, N - N // 2))
    solve(prog, x, trajs2, costs2)
    prog.SetInitialGuess(x[:N // 2], np.linspace(x0, np.array([1, 1]), N // 2))
    prog.SetInitialGuess(x[N // 2:], np.linspace(np.array([1, 1]), xf, N - N // 2))
    solve(prog, x, trajs2, costs2)

    return np.stack(trajs1+trajs2), np.array(costs1+costs2)
