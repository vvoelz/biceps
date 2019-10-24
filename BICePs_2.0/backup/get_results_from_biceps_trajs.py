#!/usr/bin/env py2

import numpy as np
import glob,os

def write_logfile(data_path, out, verbose=False):
    # Header for log file:
    log_header = "%s, %s, %s, %s, %s, %s, %s, %s\n"%("d_*", "steps", "MAX(sigma)",
        "MAX(gamma)", "Accepted([sigma,gamma,state])", "Max Population(State)", "BF", "Pass or Fail")
    if verbose:
        print(log_header)
    trajectories = glob.glob(data_path+"/results_ref_normal*/"+"traj_lambda*")
    #print(trajectories[60])
    #exit()
    with open(out, 'w') as file:
        file.seek(0)
        file.writelines("%s"%log_header)
        #for traj in trajectories[50:51]:
        for traj in trajectories[60:61]:
            traj_name,results_,d_ = traj.split("/")[-1],traj.split("/")[-2],traj.split("/")[-3]
            steps = results_.split("_")[-1]
            t = np.load(traj)["arr_0"].item()
            rest_type,sampled_parameters = t["rest_type"],np.array(t["sampled_parameters"])
            sep_accept,allowed_parameters = np.array(t["sep_accept"][0]),np.array(t["allowed_parameters"])
            max_para = [allowed_parameters[i][int(np.where(sampled_parameters[i] == np.max(sampled_parameters[i]))[0])] for i in range(len(rest_type))]

            # Checking if all the parameters have been sampled...
            check_if_failed = [np.where(sampled_parameters[i]==0)[0] for i in range(len(rest_type))]
            pass_or_fail = "Passed"
            if check_if_failed != []:
                pass_or_fail = "Failed"
                if verbose:
                   print("Failed!")

            res_dir = traj.split("traj_lam")[0]
            BF,max_pop,state = None,None,None
            if os.path.exists(res_dir+"BS.dat"):
                BF = np.loadtxt(res_dir+"BS.dat")[1,0]
            if os.path.exists(res_dir+"populations.dat"):
                pop = np.loadtxt(res_dir+"populations.dat")[:,1]
                max_pop = np.nanmax(pop)
                state = np.concatenate(np.where(pop == max_pop))

            line = "%s, %s, %s, %s, %s, %s(%s), %s, %s\n"%(d_,steps,max_para[0],max_para[1],
                    sep_accept,max_pop,state,BF,pass_or_fail)
            if verbose:
                print(line)
            file.writelines(line)
            file.truncate()




if __name__ == "__main__":

    # NOTE: Give a path to glob as the first agument
    #data_path,output_name = input[1],input[2]
    data_path = "/Volumes/RMR_4TB/new_sampling/d_*"
    output_name = "log.csv"
    write_logfile(data_path=data_path, out=output_name)








