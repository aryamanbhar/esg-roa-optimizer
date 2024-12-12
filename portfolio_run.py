#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         portfolio_run.py
 Description:  
 Author:       Samuel
 Date:         13/09/2021
---------------------------------
'''
import numpy as np
import pandas as pd
import time
import os
from benchmark_algo.de import DE
from benchmark_algo.ga import GA
from benchmark_algo.pso import PSO
from benchmark_algo.woa import WOA
from benchmark_algo.jade import JADE
from benchmark_algo.shade import SHADE
from benchmark_algo.lshade import LSHADE
from benchmark_algo.lshade_rsp import LSHADE_RSP
from benchmark_algo.nl_shade_rsp import NL_SHADE_RSP
from benchmark_algo.nl_shade_lbc import NL_SHADE_LBC
from benchmark_algo.heboalgo import HEBOALGO
from benchmark_algo.lssp import LSSP
from benchmark_algo.ea4eig import EA4EIG
from benchmark_algo.disgsa import DISGSA

from ampo.ampo import AMPO
from csde.csde import CSDE
# from de_gls_bound_adapt.de_gls_scheme5Y_adapt import DE_GLS_SCHEME5Y_ADAPT
from sade.sade import SADE
from hypo.hypo import HYPO
from ada_guide.ada_guide import ADAGUIDE
# from ada_guide.ada_guide_1 import DE_GLS_ADAGUIDE_1
# from ada_guide.ada_guide_with3ops import DE_GLS_ADAGUIDE_2
from problemSet import *
from animation import convergence_graph

np.random.seed(3407) # 3407

def run_process(task_info):
    algo_dicts = {'GA': GA, 'DE': DE, 'PSO': PSO, 'WOA': WOA, 'AMPO': AMPO, 'CSDE': CSDE, 'SADE': SADE,
                    # 'BI-AMPO-SCHEME1': BiAMPO1, 'BI-AMPO-SCHEME2': BiAMPO2,
                    # 'BI-AMPO-SCHEME3': BiAMPO3,
                    # 'ISOLATE-AMPO-CSDE': IsolateAMPOAndCSDE,
                    # 'UNCOND-BI': UncondBi,
                    # 'UNCOND-UNI-AMPO-CSDE': UncondUniAMPOToCSDE,
                    # 'UNCOND-UNI-CSDE-AMPO': UncondUniCSDEToAMPO,

                    'JADE': JADE,
                    'SHADE': SHADE,
                    'LSHADE': LSHADE,
                    'LSHADE-RSP': LSHADE_RSP,
                    'NL-SHADE-RSP': NL_SHADE_RSP,
                    'NL-SHADE-LBC': NL_SHADE_LBC,
                    'HEBO': HEBOALGO,
                    'LSSP': LSSP,
                    'EA4EIG': EA4EIG,
                    'DISGSA': DISGSA,

                    # 'DE-GLS-ADAPT-ADAGUIDE-1': DE_GLS_ADAGUIDE_1,
                    # 'DE-GLS-ADAPT-ADAGUIDE-2': DE_GLS_ADAGUIDE_2,
                    'HYPO': HYPO, # HYPO
                    'ADAGUIDE': ADAGUIDE, # ADAGUIDE
                    }

    algo_gls_lst = []
    algo_adapt_lst = []
    algo_sade_lst = []
    algo_benchmark_lst = ['GA', 'PSO', 'DE', 'ACO', 'WOA', 'JADE', 'SHADE', 'LSHADE', 'LSHADE-RSP', 'NL-SHADE-RSP', 'NL-SHADE-LBC', 'HEBO', 'LSSP', 'EA4EIG', 'DISGSA']
    algo_bicompare_lst = ['BI-AMPO-SCHEME3', 'ISOLATE-AMPO-CSDE', 'UNCOND-BI', 'UNCOND-UNI-AMPO-CSDE', 'UNCOND-UNI-CSDE-AMPO']
    for algo_n in algo_dicts.keys():
        if ('GLS' in algo_n) or ('ADAGUIDE' in algo_n):
            algo_gls_lst.append(algo_n)
        if ('ADAPT' in algo_n) or ('ADAGUIDE' in algo_n):
            algo_adapt_lst.append(algo_n)
        if 'SADE' in algo_n:
            algo_sade_lst.append(algo_n)

    run_times = task_info['run_times']
    pop = task_info['pop']
    iterations = task_info['iterations']
    verbose = task_info['verbose']

    problem_size = task_info['problem_size']
    name = task_info['algo']
    Algo = algo_dicts[task_info['algo']]
    func = task_info['func']
    final_eval_func = task_info['final_eval']
    func_name = task_info['func_name']
    bound = task_info['bound']
    root_f = task_info['root_f']
    version = task_info['version']
    err_acc = task_info['err_acc']
    optimal_value = task_info['optimal']
    func_eval = task_info['func_eval']

    cpu_times_list = []
    run_times_list = []
    time_opt_list = []
    best_fit_lst = []
    fitness_history_norm_lst = []
    fitness_history_lst = []
    best_solution_lst = []

    objective_best_fit_lst = []
    objective_fitness_history_lst = []
    objective_best_solution_lst = []
    utils_cum_lst = []

    best_sr_lst = [] # sharpe ratio
    best_return_lst = [] # returns
    best_risk_lst = [] # risk

    success_run = 0
    fit_history = []

    iter_to_opt_lst = []
    iter_to_gbest_lst = []

    penalty_factor_pd = pd.DataFrame()

    for time_no in range(0, run_times):

        start_processing_time = time.process_time()
        start_pref_time = time.perf_counter()
        if (name in algo_benchmark_lst) or (name in ['CSDE', 'SADE', 'AMPO', 'BI-AMPO-SCHEME1', 'HYPO']) or (name in algo_bicompare_lst):
            algo = Algo(func=func, dim=problem_size, bound=bound, max_iters=iterations, pop=pop,
                        func_name=func_name, show_info=verbose, func_eval=func_eval, error_val=err_acc, opt_value=optimal_value)
            best_solution, best_fit, fit_history = algo.run()
            obj_best_fit = best_fit
            obj_fit_history = fit_history
            obj_best_solution = best_solution
            iter_to_gbest_lst.append(algo.get_convergenceIter())
        elif (name in algo_gls_lst):
            algo = Algo(func=func, dim=problem_size, bound=bound, max_iters=iterations, pop=pop,
                        func_name=func_name, func_eval=func_eval, show_info=verbose, error_val=err_acc, opt_value=optimal_value)
            best_solution, best_fit, fit_history, obj_best_solution, obj_best_fit, obj_fit_history = algo.run()
            iter_to_gbest_lst.append(algo.get_convergenceIter())
        else:
            raise Exception('Cannot find the algorithm [{}]'.format(name))
        end_processing_time = time.process_time()
        end_pref_time = time.perf_counter()

        cpu_times_list.append(end_processing_time - start_processing_time)
        run_times_list.append(end_pref_time - start_pref_time)
        time_opt_list.append(algo.get_time_opt())

        path = os.path.join(root_f, 'history', name, 'func_{}'.format(func_name), 'dim_{}'.format(problem_size))
        os.makedirs(path, exist_ok=True)
        length_exist_files = len(os.listdir(path))
        hist_path = os.path.join(path, '{}.csv'.format(str(length_exist_files + 1)))

        utils_cum = 0
        obj_std, obj_mean, obj_best, obj_worst = 0, 0, 0, 0
        iteration_run = len(fit_history)
        if name in algo_gls_lst:
            rec = algo.get_rec()
            x_pd = {
                'iteration': np.arange(len(fit_history)), 'original_fitness': fit_history,
                'objective_fitness': obj_fit_history[:iteration_run],
                'original_gbest_op': rec['original_gbest_op'][:iteration_run],
                'objective_gbest_op': rec['objective_gbest_op'][:iteration_run],
                'utils_updateIter': rec['utils_updateIter'][:iteration_run],
                'utils_updateCum': rec['utils_updateCum'][:iteration_run],
                'penalty_value_max': np.max(rec['penalty_value_hist'][:iteration_run], axis=1),
                'penalty_value_avg': np.mean(rec['penalty_value_hist'][:iteration_run], axis=1),
                'penalty_value_min': np.min(rec['penalty_value_hist'][:iteration_run], axis=1),
            }
            utils_cum = np.sum(rec['utils_updateIter'][:iteration_run])
            add_cols = []
            op_size_rec = ['{}_pop_size'.format(i) for i in rec['op_list']]
            cr_mean_rec = ['cr_mean_{}'.format(i) for i in rec['op_list']]
            add_features = ['bound_revision'] + op_size_rec + cr_mean_rec
            for add_feature in add_features:
                if add_feature in rec.keys():
                    x_pd[add_feature] = rec[add_feature][:iteration_run]
                    add_cols.append(add_feature)
                    if add_feature == 'bound_revision':
                        rev_iter = np.where(np.array(rec['bound_revision'][:iteration_run]) == "", 0, 1)
                        x_pd['bound_revision_cum'] = np.cumsum(rev_iter, axis=0)
                        add_cols.append('bound_revision_cum')
            x_pd_cols = ['iteration', 'original_fitness', 'objective_fitness', 'original_gbest_op', 'objective_gbest_op',
                        'utils_updateIter', 'utils_updateCum']
            x_pd_cols = x_pd_cols + add_cols
            x_pd = pd.DataFrame(x_pd, columns=x_pd_cols)
            x_pd.to_csv(hist_path, index=False)

            # path_penalty_factor = os.path.join(root_f, 'penalty_factor')
            # os.makedirs(path_penalty_factor, exist_ok=True)
            
            # # penalty factor, Not applicable for bound revision scheme.
            # path_penalty_factor = os.path.join(path_penalty_factor, 'algo_{}_func_{}_dim_{}.csv'.format(name, func_name, problem_size))
            # penalty_factor_mtx = pd.DataFrame(rec['penalty_factor'])
            # penalty_factor_pd = pd.concat([penalty_factor_pd, penalty_factor_mtx], axis=0, join='outer')
            # penalty_factor_pd.reset_index(drop=True, inplace=True)
            # penalty_factor_pd.loc[len(penalty_factor_pd), :] = [np.inf] * rec['dimInterval']
            # penalty_factor_pd.to_csv(path_penalty_factor, index=False)

        elif name == 'SADE':
            rec = algo.get_rec()
            x_pd = {
                'iteration': np.arange(len(fit_history)), 'original_fitness': fit_history,
                'original_gbest_op': rec['original_gbest_op'][:iteration_run],
            }
            add_cols = []
            op_size_rec = ['{}_pop_size'.format(i) for i in rec['op_list']]
            add_features = op_size_rec
            for add_feature in add_features:
                if add_feature in rec.keys():
                    x_pd[add_feature] = rec[add_feature][:iteration_run]
                    add_cols.append(add_feature)
            x_pd_cols = ['iteration', 'original_fitness', 'original_gbest_op']
            x_pd_cols = x_pd_cols + add_cols
            x_pd = pd.DataFrame(x_pd, columns=x_pd_cols)
            x_pd.to_csv(hist_path, index=False)
        
        elif (name in ['AMPO', 'BI-AMPO-SCHEME1', 'HYPO']) or (name in algo_bicompare_lst):
            rec = algo.get_rec()
            x_pd = {'iteration': np.arange(len(fit_history)), 'DE': algo.rec_de[:iteration_run],
                    'recovery_pop_size': algo.rec_recovery[:iteration_run], 'recovery_cum_number': algo.op_stat['recovery_cum_number'][:iteration_run], 'fitness': fit_history, 'gbest_op': algo.gbest_op[:iteration_run], 'local_pop_size': algo.op_stat['local_search'][:iteration_run], 'global_pop_size': algo.op_stat['global_search'][:iteration_run],
                    'random_pop_size': algo.op_stat['random_search'][:iteration_run], 'leader_pop_size': algo.op_stat['leader'][:iteration_run], 'local_recovery_pop_size': algo.op_stat['local_recovery'][:iteration_run], 'global_recovery_pop_size': algo.op_stat['global_recovery'][:iteration_run], 'leader_recovery_pop_size': algo.op_stat['leader_recovery'][:iteration_run], 'DE_pop_size': algo.op_stat['DE'][:iteration_run], 'global_search_step_size': algo.step_size['global_search'][:iteration_run],
                    'local_search_step_size': algo.step_size['local_search'][:iteration_run], 'DE_step_size': algo.step_size['DE'][:iteration_run], 'random_search_step_size': algo.step_size['random_search'][:iteration_run], 'gbestStep_step_size': algo.step_size['gbestStep'][:iteration_run]
                    }

            x_pd_cols = ['iteration', 'DE', 'recovery_pop_size', 'recovery_cum_number', 'fitness', 'gbest_op', 'local_pop_size', 'global_pop_size', 'random_pop_size', 'leader_pop_size',
                        'local_recovery_pop_size', 'global_recovery_pop_size', 'leader_recovery_pop_size', 'DE_pop_size', 'global_search_step_size', 'local_search_step_size', 'DE_step_size',
                        'random_search_step_size', 'gbestStep_step_size']
            global_search = algo.step_size['global_search'][:iteration_run]
            local_search = algo.step_size['local_search'][:iteration_run]
            de_search = algo.step_size['DE'][:iteration_run]
            random_search = algo.step_size['random_search'][:iteration_run]

            # print("Global_Step_min: {}, Local_min: {}, DE_Step_min: {}, Random_min: {}".format(
                # np.min(global_search[global_search > 0]), np.min(local_search[local_search > 0]), np.min(de_search[de_search > 0]), np.min(random_search[random_search > 0])))
            x_pd = pd.DataFrame(x_pd, columns=x_pd_cols)
            x_pd.to_csv(hist_path, index=False)

        elif name == 'CSDE':
            rec = algo.get_rec()
            # recovery_cum_number pending
            x_pd = {'iteration': np.arange(len(fit_history)), 'fitness': fit_history, 'gbest_op': algo.gbest_op[:iteration_run], 'CurrentToPbest_pop_size': algo.op_stat['CurrentToPbest'][:iteration_run], 'PbestToRand_pop_size': algo.op_stat['PbestToRand'][:iteration_run],
                    'CurrentToPbest_step_size': algo.step_size['CurrentToPbest'][:iteration_run], 'CurrentToPbest_F': algo.F_size['CurrentToPbest'][:iteration_run], 'PbestToRand_step_size': algo.step_size['PbestToRand'][:iteration_run], 'PbestToRand_F': algo.F_size['PbestToRand'][:iteration_run],
                    'gbestStep_step_size': algo.step_size['gbestStep'][:iteration_run], 'gbestStep_F': algo.F_size['gbestStep'][:iteration_run]}
            # print("Fgc_Step_min: {}, Fgc_min: {}, Fgr_Step_min: {}, Fgr_min: {}".format(
            #     np.min(algo.step_size['CurrentToPbest'][:iteration_run][algo.step_size['CurrentToPbest'][:iteration_run] > 0]), np.min(algo.F_size['CurrentToPbest'][:iteration_run][algo.F_size['CurrentToPbest'][:iteration_run] > 0]),
            #     np.min(algo.step_size['PbestToRand'][:iteration_run][algo.step_size['PbestToRand'][:iteration_run] > 0]), np.min(algo.F_size['PbestToRand'][:iteration_run][algo.F_size['PbestToRand'][:iteration_run] > 0])))

            x_pd = pd.DataFrame(x_pd, columns=['iteration', 'fitness', 'gbest_op', 'CurrentToPbest_pop_size', 'PbestToRand_pop_size', 'CurrentToPbest_step_size', 'CurrentToPbest_F',
                                                'PbestToRand_step_size', 'PbestToRand_F', 'gbestStep_step_size', 'gbestStep_F'])
            x_pd.to_csv(hist_path, index=False)

        else:
            x_pd = {'iteration': np.arange(len(fit_history)), 'fitness': fit_history}
            x_pd = pd.DataFrame(x_pd, columns=['iteration', 'fitness'])
            x_pd.to_csv(hist_path, index=False)


        best_fit_lst.append(best_fit)
        fit_history_norm = np.array(fit_history) - optimal_value
        fitness_history_norm_lst.append(fit_history_norm) # TODO
        fitness_history_lst.append(fit_history)
        best_solution_lst.append(best_solution)

        objective_best_fit_lst.append(obj_best_fit)
        objective_fitness_history_lst.append(obj_fit_history)
        objective_best_solution_lst.append(obj_best_solution)
        utils_cum_lst.append(utils_cum)

        # For PO only
        sig_fit, sig_sr, sig_returns, sig_risk = final_eval_func(weights=best_solution)
        best_sr_lst.append(sig_sr)
        best_return_lst.append(sig_returns)
        best_risk_lst.append(sig_risk)
        sr_mean = np.mean(best_sr_lst)
        sr_std = np.std(best_sr_lst)
        sr_best = np.max(best_sr_lst)
        sr_worst = np.min(best_sr_lst)
        
        returns_mean = np.mean(best_return_lst)
        returns_std = np.std(best_return_lst)
        returns_best = np.max(best_return_lst)
        returns_worst = np.min(best_return_lst)

        risk_mean = np.mean(best_risk_lst)
        risk_std = np.std(best_risk_lst)
        risk_best = np.min(best_risk_lst)
        risk_worst = np.max(best_risk_lst)

        print("name: {}, func: {}, dim: {}, time: {}/{}, total runs: {}, fit: {}, sr: {}, returns: {}, risk: {}, Conv: {}, pop: {}, func_eval/pop: {}, func_eval: {}"
                .format(name,func_name, problem_size, time_no + 1, run_times, length_exist_files+1, best_fit, sig_sr, sig_returns, sig_risk, 
                        algo.get_convergenceIter(), pop, iterations, pop*iterations))
        print("-----------------------------------")

        is_success_flag = False
        if (best_fit - optimal_value) <= err_acc:
            success_run = success_run + 1
            is_success_flag = True
            # 只统计找到opt的runtime
            d_x1 = np.where(fit_history_norm <= err_acc, 0, 1)
            d_x2 = np.sum(d_x1)
            if d_x2 >= len(d_x1):
                # Non-convergence in the experiment
                curIter_to_opt = len(d_x1)
            else:
                # Convergence
                curIter_to_opt = d_x2 + 1
            iter_to_opt_lst.append(curIter_to_opt)

        std = np.std(best_fit_lst)
        mean = np.mean(best_fit_lst)
        best = np.min(best_fit_lst)
        worst = np.max(best_fit_lst)

        # obj_std = np.std(objective_best_fit_lst)
        # obj_mean = np.mean(objective_best_fit_lst)
        # obj_best = np.min(objective_best_fit_lst)
        # obj_worst = np.max(objective_best_fit_lst)

        avg_cpu_time = np.mean(cpu_times_list)
        avg_run_time = np.mean(run_times_list)
        avg_opt_time = np.mean(time_opt_list)
        if len(iter_to_opt_lst) == 0:
            iter_to_opt = 0
        else:
            iter_to_opt = np.mean(iter_to_opt_lst)

        if len(iter_to_gbest_lst) == 0:
            iter_to_gbest = 0
        else:
            if is_success_flag:
                iter_to_gbest_lst[-1] = curIter_to_opt
                iter_to_gbest = np.mean(iter_to_gbest_lst)
            else:
                iter_to_gbest = np.mean(iter_to_gbest_lst)
        podir = os.path.join(root_f, 'po')
        os.makedirs(podir, exist_ok=True)
        po_path = os.path.join(podir, 'algo_{}_func_{}_dim_{}.csv'.format(name, func_name, problem_size))
        if os.path.exists(po_path):
            po_existing_data = pd.read_csv(po_path, header=0)
            if len(po_existing_data) != length_exist_files:
                raise ValueError("The existing po.csv file has {} record. But the number of history files is {} | path: {}".format(len(po_existing_data), length_exist_files, po_path))
            
            hist_best_sr_lst = po_existing_data['sr'].values
            cur_best_sr_lst = hist_best_sr_lst
            cur_best_sr_lst.append(sig_sr)
            sr_mean = np.mean(cur_best_sr_lst)
            sr_std = np.std(cur_best_sr_lst)
            sr_best = np.max(cur_best_sr_lst)
            sr_worst = np.min(cur_best_sr_lst)

            hist_best_return_lst = po_existing_data['returns'].values
            cur_best_return_lst = hist_best_return_lst
            cur_best_return_lst.append(sig_returns)
            returns_mean = np.mean(cur_best_return_lst)
            returns_std = np.std(cur_best_return_lst)
            returns_best = np.max(cur_best_return_lst)
            returns_worst = np.min(cur_best_return_lst)

            hist_best_risk_lst = po_existing_data['risk'].values
            cur_best_risk_lst = hist_best_risk_lst
            cur_best_risk_lst.append(sig_risk)
            risk_mean = np.mean(cur_best_risk_lst)
            risk_std = np.std(cur_best_risk_lst)
            risk_best = np.min(cur_best_risk_lst)
            risk_worst = np.max(cur_best_risk_lst)

        else:
            cur_best_sr_lst = best_sr_lst
            cur_best_return_lst = best_return_lst
            cur_best_risk_lst = best_risk_lst

        path = os.path.join(root_f, 'running')
        os.makedirs(path, exist_ok=True)
        x_path = os.path.join(path, 'algo_{}_func_{}_dim_{}.csv'.format(name, func_name, problem_size))
        if os.path.exists(x_path): 
            existing_data = pd.read_csv(x_path, header=0)
            if len(existing_data) != 1:
                raise ValueError("The existing running.csv file has more than one record. {} | path: {}".format(len(existing_data), x_path))
            hist_runs = existing_data['num_of_cur_iter'].values[0]
            hist_best_fit_lst = list(eval(existing_data['best_fit_list'].values[0]))
            acc_runs = hist_runs + 1 # num_of_cur_iter
            hmean = existing_data['fit_mean'].values[0]
            cur_best_fit_lst = hist_best_fit_lst
            cur_best_fit_lst.append(best_fit_lst[-1])
            std = np.std(cur_best_fit_lst)
            mean = ((hmean * hist_runs) + best_fit_lst[-1]) / acc_runs
            best = np.min(cur_best_fit_lst)
            worst = np.max(cur_best_fit_lst)
            cur_avg_cpu_time = ((existing_data['avg_cpu_time'].values[0] * hist_runs) + cpu_times_list[-1]) / acc_runs
            cur_avg_run_time = ((existing_data['avg_run_time'].values[0] * hist_runs) + run_times_list[-1]) / acc_runs
            cur_avg_opt_time = ((existing_data['avg_opt_time'].values[0] * hist_runs) + time_opt_list[-1]) / acc_runs
            if is_success_flag:
                cur_success_run = existing_data['success_run'].values[0] + 1
                cur_iter_to_opt = ((existing_data['iterations_to_optimal'].values[0] * (cur_success_run-1)) + curIter_to_opt) / cur_success_run
            else:
                cur_success_run = existing_data['success_run'].values[0]
                cur_iter_to_opt = existing_data['iterations_to_optimal'].values[0]
            
            cur_iter_to_gbest = ((existing_data['iterations_to_gbest'].values[0] * hist_runs) + iter_to_gbest_lst[-1]) /acc_runs
            cur_utils_cum = ((existing_data['utils_cum'].values[0] * hist_runs) + utils_cum) /acc_runs
        
        else:
            if len(best_fit_lst) != 1:
                raise ValueError("The best_fit_lst has more than one record. {} | path: {}".format(len(best_fit_lst), x_path))
            acc_runs = 1 # num_of_cur_iter
            cur_best_fit_lst = best_fit_lst
            cur_avg_cpu_time = avg_cpu_time
            cur_avg_run_time = avg_run_time
            cur_avg_opt_time = avg_opt_time
            cur_success_run = success_run
            cur_iter_to_opt = iter_to_opt
            cur_iter_to_gbest = iter_to_gbest
            cur_utils_cum = np.mean(utils_cum_lst)        
        
        out = {
            'function': str(func_name),
            'version': version,
            'dim': problem_size,
            'algo': name,
            'num_of_cur_iter': acc_runs,
            'optimal': optimal_value,
            'sr_mean': sr_mean,
            'sr_std': sr_std,
            'sr_best': sr_best,
            'sr_worst': sr_worst,
            'returns_mean': returns_mean,
            'returns_std': returns_std,
            'returns_best': returns_best,
            'returns_worst': returns_worst,
            'risk_mean': risk_mean,
            'risk_std': risk_std,
            'risk_best': risk_best,
            'risk_worst': risk_worst,
            'fit_mean': mean,
            'fit_std': std,
            'fit_best': best,
            'fit_worst': worst,
            'avg_cpu_time': cur_avg_cpu_time,
            'avg_run_time': cur_avg_run_time,
            'avg_opt_time': cur_avg_opt_time,
            'success_run': cur_success_run,
            'iterations_to_optimal': cur_iter_to_opt, # should be divided by 50 if pop = 1
            'iterations_to_gbest': cur_iter_to_gbest,
            'best_fit_list': cur_best_fit_lst
        }
        #
        pd.DataFrame([out]).to_csv(x_path, index=False)

        # Solution
        # path = os.path.join(root_f, 'solution')
        # os.makedirs(path, exist_ok=True)
        # # Record the best solution of each runtime
        # pd.DataFrame(best_solution_lst).to_csv(os.path.join(path, 'algo_{}_func_{}_dim_{}_bestsolution.csv'.format(name, func_name, problem_size)), index=False)
    
        # if name not in algo_benchmark_lst:
        #     pd.DataFrame(objective_best_solution_lst).to_csv(os.path.join(path, 'objetive_algo_{}_func_{}_dim_{}_bestsolution.csv'.format(name, func_name, problem_size)), index=False)

        # For PO only
 
        po_data = {
            'sr': cur_best_sr_lst, # sharpe ratio
            'returns': cur_best_return_lst, # returns
            'risk': cur_best_risk_lst,# risk
            }
        po_data = pd.DataFrame(po_data, columns=['sr', 'returns', 'risk'])
        po_data.to_csv(po_path, index=False)

        # 'version/'
        # running:/all
        # history/algo/func_xx/dim_x/iter_xx.csv
        # convergence: func_cec2_dim_30_times_10_iter_1000.png
        # best_solution/
        # solution
        # version, dim, algo, func
        try:
        # if True:
            # convergence_graph(fitness_lst=fitness_history_norm_lst, func_name=func_name, problem_size=problem_size,
            #                     run_times=run_times, iterations=iterations, root_f=root_f, algo_name=name)

            # if name in algo_gls_lst:
            #     # 3-graph: fitness, obj_fitness, avg_compare
            #     original_objective_compare_graph(original_fit=fitness_history_lst, objective_fit=objective_fitness_history_lst, func_name=func_name, problem_size=problem_size,
            #                     run_times=run_times, iterations=iterations, root_f=root_f, algo_name=name)

            #     penalty_value_graph(fitness_norm=fitness_history_norm_lst[-1], max_p=np.max(rec['penalty_value_hist'], axis=1),
            #                         avg_p=np.mean(rec['penalty_value_hist'], axis=1),
            #                         min_p=np.min(rec['penalty_value_hist'], axis=1), func_name=func_name,
            #                         problem_size=problem_size, time_no=length_exist_files+1, iterations=iterations,
            #                         root_f=root_f, algo_name=name)
            # if (name in algo_adapt_lst) and (name not in algo_sade_lst):
            #     dataLst = algo.get_rec()
            #     paraTrend_graph(dataLst=dataLst, func_name=func_name, problem_size=problem_size,
            #                     iterations=iterations, root_f=root_f, algo_name=name, time_no=length_exist_files+1, FP=algo.FP)
            # if name in algo_sade_lst:
            #     dataLst = algo.get_rec()
            #     paraTrend_SADE(dataLst=dataLst, func_name=func_name, problem_size=problem_size, 
            #                     root_f=root_f, algo_name=name, time_no=length_exist_files+1)
            pass
        except:
            print('Cannot output the convergence graph. {}_{}_{}'.format(func_name, problem_size, length_exist_files+1))

        # del algo
        # gc.collect()

def entrance():
    """
        DOW: 29 stocks (30 in total), 1258 trading days
        HSI: 59 stocks (69 in total), 1233 trading days
        CSI300: 239 stocks (300 in total), 1213 trading days [50, 100, 150, 200, 239]
        SP500: 490 stocks (502 in total), 1258 trading days [100, 200, 300, 400, 490]
        mix_SP500_CSI300_HSI: (788 stocks), 1148 trading days [100, 200, 400, 600, 788] 
    """
    info_dict = {
        'CSI300': {'dim_list': [239], }, # [50, 100, 150, 200, 239] , iter [2k, 2k, 3k, 3k, 3k]
        'SP500': {'dim_list': [100, 200, 300, 400, 490], }, # [100, 200, 300, 400, 490], iter [2k, 3k, 4k, 5k, 6k]
        'mix_SP500_CSI300_HSI': {'dim_list': [100, 200, 400, 600, 788], }, # [100, 200, 400, 600, 788], iter [2k, 3k, 5k, 7k, 9k]
    }

    # dim, market, algo
    market_name = ['CSI300'] # ['CSI300', 'SP500', 'mix_SP500_CSI300_HSI'] # ['CSI300', 'SP500', 'mix_SP500_CSI300_HSI'] # 'CSI300', 'SP500', 'mix_SP500_CSI300_HSI'
    # BI-AMPO-SCHEME3, ISOLATE-AMPO-CSDE, UNCOND-BI, UNCOND-UNI-AMPO-CSDE, UNCOND-UNI-CSDE-AMPO
    # JADE, SHADE, LSHADE, LSHADE-RSP, NL-SHADE-RSP, NL-SHADE-LBC, HEBO, LSSP, EA4EIG, DISGSA
    # algo_lst = ['BI-AMPO-SCHEME3', 'ISOLATE-AMPO-CSDE', 'UNCOND-BI', 'UNCOND-UNI-AMPO-CSDE', 'UNCOND-UNI-CSDE-AMPO'] # AMPO, CSDE, DE, GA, PSO, SADE, WOA, BI-AMPO-SCHEME1, BI-AMPO-SCHEME2, DE-GLS-SCHEME5Y-ADAPT
    algo_lst = ['ADAGUIDE'] # ['HYPO', 'NL-SHADE-RSP']

    version = 'v87'
    err_acc = 10e-8
    data_dir = 'stock_data'

    run_times = 30 # 30
    pop = 50 #

    verbose = False
    task_cnt = 0
    for mkt_name in market_name:
        dim_lst = info_dict[mkt_name]['dim_list']
        for algo in algo_lst:
            for problem_size in dim_lst:
                if problem_size >= 600:
                    enable_torch = False
                else:
                    enable_torch = False
                po_instance = PortfolioOpt(data_dir=data_dir, market_name=mkt_name, dim=problem_size, torch_flag=enable_torch)
                func = po_instance.func
                final_eval = po_instance.final_eval
                root_f = os.path.join('res_{}_PO'.format(version), mkt_name, 'resPO_{}_{}_dim{}'.format(version, algo, problem_size))
                os.makedirs(root_f, exist_ok=True)
                task_cnt = task_cnt + 1
                k = int(np.floor((problem_size/100) + 0.5)) + 1
                func_eval = int(50000 * k) # Iter: 1000 * k
                iterations = int(np.ceil(func_eval / pop)) #
                if algo in ['HEBO', 'LSSP']:
                    run_times_adj = 5
                else:
                    run_times_adj = run_times
                task_info = {
                    'Task_ID': task_cnt,
                    'func': func,
                    'final_eval': final_eval,
                    'algo': algo,
                    'problem_size': problem_size,
                    'root_f': root_f,
                    'version': version,
                    'err_acc': err_acc,
                    'run_times': run_times_adj,
                    'pop': pop,
                    'iterations': iterations,
                    'func_eval': func_eval,
                    'verbose': verbose,
                    'func_name': mkt_name,
                    'bound': [-1, 1],
                    'optimal': 0,
                }
                run_process(task_info=task_info)

def main():
    entrance()

if __name__ == '__main__':
    main()
    
    
    