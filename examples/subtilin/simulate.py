"""
An example for subtilin production model
"""
import argparse
import copy
import functools
import importlib
import os
from enum import Enum
from logging import getLogger
from multiprocessing import Process, Lock
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use('Agg')
import pandas as pd
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm

from deep_sentinel.utils import mkdir, parallel

if TYPE_CHECKING:
    from typing import List, Tuple, Optional

logger = getLogger(__name__)

# Constants
r = 0.02
D_max = 1
k_1 = 0.1
k_2 = 0.4
k_3 = 0.5
k_4 = 1
k_5 = 1
xi = 0.1
lambda_1 = lambda_2 = lambda_3 = 0.2
eta = 4
# e^{\frac{-\delta G_{rk}}{RT}}
e_delta_grk = 0.4
# e^{\frac{-\delta G_{s}}{RT}}
e_delta_gs = 0.4
# Initial values
# [SigH] = [SpaRK] = [SpaS] = 0
# [SigH] means a concentration of SigH in a cell.
D_0 = 0.01
sigh_0 = spark_0 = spas_0 = 0
X_0 = 10

keys = ['D', 'X', '[SigH]', '[SpaRK]', '[SpaS]']
prefix = 'subtilin'
step = 1

_lock = Lock()


class State(Enum):
    DISABLED = 0
    ENABLED = 1

    def is_enabled(self) -> bool:
        if self.name == self.ENABLED.name:
            return True
        return False

    @classmethod
    def get_choices(cls):
        return tuple(t.value for t in cls)


class MarkovChain(object):
    available_states = [State.DISABLED, State.ENABLED]

    def __init__(self, initial_transition_matrix: 'np.ndarray'):
        self.transitions = []
        self.transitions.append(self._gen_state(initial_transition_matrix))

    def next_state(self, transition_matrix: 'np.ndarray') -> State:
        state = self._gen_state(transition_matrix)
        self.transitions.append(state)
        return state

    def _gen_state(self, transition_matrix: 'np.ndarray') -> State:
        return np.random.choice(
            self.available_states, p=transition_matrix[self.current_state.value]
        )

    @property
    def current_state(self) -> 'State':
        if len(self.transitions) == 0:
            return State.DISABLED
        return self.transitions[-1]


class BacillusSubtilis(object):
    nutrients_threshold = eta * D_max

    def __init__(self, sig_h: float, spa_rk: float, spa_s: float):
        self.sig_h = sig_h
        self.spark = spa_rk
        self.spas = spa_s
        self.s1_chain = MarkovChain(self.transition_matrix_A)
        self.s2_chain = MarkovChain(self.transition_matrix_B)

    def grow(self, x: 'np.float') -> 'List[np.float]':
        self.s1_chain.next_state(self.transition_matrix_A)
        self.s2_chain.next_state(self.transition_matrix_B)
        return [self.sig_h_dt(x), self.spa_rk_dt(), self.spa_s_dt()]

    def sig_h_dt(self, x):
        if x < self.nutrients_threshold:
            return k_3 - (lambda_1 * self.sig_h)
        return -1 * lambda_1 * self.sig_h

    def spa_rk_dt(self):
        if self.s1.is_enabled():
            return k_4 - (lambda_2 * self.spark)
        return -1 * lambda_2 * self.spark

    def spa_s_dt(self):
        if self.s2.is_enabled():
            return k_5 - (lambda_3 * self.spas)
        return -1 * lambda_3 * self.spas

    @property
    def s1(self) -> 'State':
        return self.s1_chain.current_state

    @property
    def s2(self) -> 'State':
        return self.s2_chain.current_state

    @property
    def transition_matrix_A(self):
        _sig_h = self.sig_h
        if _sig_h < 0:
            _sig_h = 0
        a0 = (e_delta_grk * _sig_h) / (1 + e_delta_grk * _sig_h)
        a1 = 1 / (1 + e_delta_grk * _sig_h)
        return np.array(
            [[1 - a0, a0],
             [a1, 1 - a1]]
        )

    @property
    def transition_matrix_B(self):
        _spark = self.spark
        if _spark < 0:
            _spark = 0
        b0 = (e_delta_gs * _spark) / (1 + e_delta_grk * _spark)
        b1 = 1 / (1 + e_delta_gs * _spark)
        return np.array(
            [[1 - b0, b0],
             [b1, 1 - b1]]
        )

    def update_concentration(self, sig_h, spa_rk, spa_s):
        self.sig_h = sig_h
        self.spark = spa_rk
        self.spas = spa_s

    @property
    def transitions(self) -> 'List[Tuple[State, State]]':
        s1_transitions = self.s1_chain.transitions
        s2_transitions = self.s2_chain.transitions
        return list(zip(s1_transitions, s2_transitions))


class Incubator(object):

    def __init__(self):
        self.b_subtilis = BacillusSubtilis(sigh_0, spark_0, spas_0)

    def incubate(self, t, x, params):
        _D = x[0]
        nutrients = x[1]
        D_inf = min(nutrients / 4, D_max)
        D_dt = r * _D * (1 - _D / D_inf)
        dt_dx = -k_1 * _D + k_2 * self.b_subtilis.spas * xi
        self.b_subtilis.update_concentration(*x[2:])
        return [D_dt, dt_dx, *self.b_subtilis.grow(nutrients)]


def solve(solver, end, index_start: int = 0):
    df_set = []
    while solver.successful() and solver.t < end:
        solver.integrate(solver.t + step)
        index_start += step
        current_result = pd.DataFrame(solver.y[None, :], index=[index_start], columns=keys)
        current_result.index.name = 'index'
        df_set.append(current_result)
        logger.debug(current_result)
    return pd.concat(df_set)


class Simulator(object):

    def __init__(self,
                 output_dir: 'Path',
                 integrator: str,
                 integrate_method: str,
                 nsteps: int,
                 train_length: int,
                 seed_length: int,
                 trial_length: int,
                 total_seeds: int,
                 total_trials: int):
        """
        Simulator is a wrapper to generate data.
        :param output_dir: Path to output directory
        :param integrator: Integrator name to pass scipy.ode
        :param integrate_method: Solver name to pass scipy.ode
        :param nsteps: Number of steps to solve
        :param train_length: Length of generated train data
        :param seed_length: Length of generated seed data
        :param trial_length: Length of generated trial data
        :param total_seeds: Number of seed data to generate
        :param total_trials: Number of trial data to generate
        """
        self.output_dir = mkdir(output_dir)

        # ODE settings
        self.ode_setting = {
            "integrator": integrator,
            "method": integrate_method,
            "nsteps": nsteps,
        }
        self.train_length = train_length
        self.seed_length = seed_length
        self.trial_length = trial_length
        self.total_seeds = total_seeds
        self.total_trials = total_trials

    def _create_solver(self,
                       incubator: 'Optional[Incubator]' = None,
                       initial_values=((D_0, X_0, sigh_0, spark_0, spas_0), 0.0)):
        if incubator is None:
            incubator = self._create_incubator()
        solver = integrate.ode(incubator.incubate).set_integrator(
            name=self.ode_setting['integrator'],
            method=self.ode_setting['method'],
            nsteps=self.ode_setting['nsteps']
        )
        solver.set_initial_value(*initial_values)
        solver.set_f_params({})
        return solver

    def _create_incubator(self) -> 'Incubator':
        return Incubator()

    def seed_indexes(self):
        return list(range(1, self.total_seeds + 1))

    def simulation_indexes(self):
        return list(range(1, self.total_trials + 1))

    def generate_train_data(self):
        incubator = self._create_incubator()
        solver = self._create_solver(incubator)
        train_data = self.output_dir / f"{prefix}-training.csv"
        df = solve(solver, self.train_length)
        ax = df.plot(subplots=True, layout=(len(keys), 1), title='Subtilin production', figsize=(14, 7))
        ax[0][0].get_figure().savefig(f'{self.output_dir / "plot.png"}')
        df.to_csv(str(train_data))
        logger.info(f"Save training data as '{train_data}'")

    def generate_simulation_data(self,
                                 args: 'Tuple[int, Incubator]',
                                 seed_number: int,
                                 initial_values: tuple):
        # Reload module to clear state
        importlib.reload(integrate)
        np.random.seed()

        simulation_number, incubator = args
        start_at = self.seed_length
        end_at = start_at + self.trial_length

        simulation_dir = self.output_dir / str(seed_number) / 'simulations'
        simulation_data = simulation_dir / f"{prefix}-{simulation_number}.csv"
        with _lock:
            if not simulation_dir.exists():
                simulation_dir.mkdir(parents=True)

        solver = self._create_solver(incubator, (initial_values, start_at))
        df = solve(solver, end_at, index_start=start_at)
        df.to_csv(str(simulation_data))

    def generate_seed_data(self, seed_number: int):
        seed_data = self.output_dir / str(seed_number) / f"{prefix}-seed.csv"
        with _lock:
            if not seed_data.parent.exists():
                seed_data.parent.mkdir(parents=True)

        incubator = self._create_incubator()
        solver = self._create_solver(incubator)

        df = solve(solver, self.seed_length)
        df.to_csv(str(seed_data))
        return df.iloc[-1].values.tolist(), incubator


def simulate(args):
    n_jobs = args.n_jobs
    if n_jobs < 0:
        n_jobs = os.cpu_count()

    simulator = Simulator(
        args.output_dir,
        args.integrator,
        args.integrate_method,
        args.nsteps,
        args.train_length,
        args.seed_length,
        args.trial_length,
        args.total_seeds,
        args.total_trials
    )
    logger.debug(f"Selected ODE settings: {simulator.ode_setting}")

    # Generate training dataset process
    train_gen_process = Process(target=simulator.generate_train_data)
    try:
        logger.info(f"Start process for generating train dataset ({simulator.train_length} steps)")
        train_gen_process.start()
        if n_jobs == 1:
            train_gen_process.join()
            # Reload module to clear state
            importlib.reload(integrate)

        # Reset seed
        np.random.seed()

        # Generate seed and simulation dataset
        logger.info(f"Start to simulate {simulator.seed_length} steps"
                    f" for each seed (total {simulator.total_seeds} seeds)")
        logger.info(f"After that, simulate {simulator.trial_length} steps"
                    f" for each seed ({simulator.total_trials} times)")
        seed_indexes = simulator.seed_indexes()
        seed_results = parallel(simulator.generate_seed_data,
                                seed_indexes,
                                n_jobs if len(seed_indexes) > n_jobs else len(seed_indexes))
        for i, result in tqdm(list(enumerate(seed_results)), desc="Completed seeds", unit="seed", leave=False):
            initial_values, incubator = result
            sim_func = functools.partial(simulator.generate_simulation_data,
                                         seed_number=i+1,
                                         initial_values=initial_values)
            sim_indexes = simulator.simulation_indexes()
            parallel(sim_func,
                     [(k, copy.deepcopy(incubator)) for k in sim_indexes],
                     n_jobs if len(sim_indexes) > n_jobs else len(sim_indexes))
        train_gen_process.join()
    except KeyboardInterrupt:
        logger.error("Ctrl + C detected. Shutdown workers.")
        train_gen_process.terminate()
    except Exception as e:
        train_gen_process.terminate()
        logger.exception(e)
        return 1
    finally:
        train_gen_process.join()
    logger.info(f"Finished")
    return 0


def get_parser():
    parser = argparse.ArgumentParser(description="An example for subtilin production model")
    parser.add_argument("-o",
                        "--output-dir",
                        type=Path,
                        help="Path to output dir",
                        required=True)
    parser.add_argument("-n",
                        "--n-jobs",
                        type=int,
                        default=-1,
                        help="Number of CPU cores to use (negative value indicate all cores)")
    ode_parser = parser.add_argument_group("ODE")
    ode_parser.add_argument("-i",
                            "--integrator",
                            type=str,
                            choices=['vode', 'zvode', 'isoda'],
                            default='vode',
                            help="Integrator name")
    ode_parser.add_argument("-m",
                            "--integrate-method",
                            type=str,
                            choices=['adams', 'bdf'],
                            default='bdf',
                            help='Which solver to use')
    ode_parser.add_argument('--nsteps',
                            type=int,
                            default=100000,
                            help='Maximum number of internally defined steps of solver')
    sim_params = parser.add_argument_group('Simulation params')
    sim_params.add_argument('--train-length',
                            type=int,
                            default=100000,
                            help='Length of training data')
    sim_params.add_argument('--total-seeds',
                            type=int, default=1, help="Number of seed data")
    sim_params.add_argument('--seed-length',
                            type=int, default=1000, help='Length of each seed data')
    sim_params.add_argument('--total-trials',
                            type=int, default=1000, help='Number of simulation data')
    sim_params.add_argument('--trial-length',
                            type=int, default=300, help='Length of each trial data')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    return simulate(args)


if __name__ == "__main__":
    import logging
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    exit(main())
