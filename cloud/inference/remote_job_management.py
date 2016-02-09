import os
import fabric.api as fapi
import fabric.context_managers as fcm
import numpy as np

def run_as_host(cmd, host, sudo=False, **kwargs):
    with fcm.settings(host_string=host):
        if sudo:
            r = fapi.sudo(cmd, pty=False, **kwargs)
        else:
            r = fapi.run(cmd, pty=False, **kwargs)
    return r

def nohup_return_pid(cmd, host, log_prefix, sudo=False, **kwargs):
    full_cmd = "mkdir -p %s && nohup %s > %s 2> %s & echo $!" % (log_prefix, cmd,
                                                                 os.path.join(log_prefix, "out.txt"), 
                                                                 os.path.join(log_prefix, "err.txt"))
    r = run_as_host(full_cmd, host, sudo=sudo, **kwargs)
    pid = int(r.split("\n")[-1])
    #pid = 1
    return pid

def running_processes(host):
    r = run_as_host("ps ax -o pid")
    pids = [int(l) for l in r.split("\n")]
    return pids

class JobManager(object):

    """
    Manages the state of jobs running on a set of remote
    hosts. Automatically allocates jobs to machines with spare CPUs.
    """

    def __init__(self, hosts, ncpus=4, log_prefix=None):
        self.ncpus = ncpus
        self.job_counter = 0

        # maps hosts to [{pid: (jobid, job_cpus)}]
        self.jobs_by_host = dict([(host, {}) for host in hosts])

        # maps jobid to (host, pid)
        self.hosts_by_job = {}

        self.log_prefix = log_prefix

    def _host_cpus_free(self, host):
        in_use = np.sum([job_cpus for (jobid, job_cpus) in self.jobs_by_host[host].values()])
        return self.ncpus - in_use

    def get_job_output(self, jobid, stderr=False):
        host, pid = self.hosts_by_job[jobid]

        logstr = os.path.join(log_prefix, "job%04d" % jobid)
        logfile = logstr + "_err.txt" if stderr else "_out.txt" 
        
        return run_as_host(host, "cat %s" % logfile)

    def run_job(self, cmd, job_cpus=1, sudo=False, **kwargs):
        
        def get_free_host(job_cpus):
            for host in self.jobs_by_host.keys():
                if self._host_cpus_free(host) >= job_cpus:
                    return host
            raise Exception("job requires %d cpus but no host has spare capacity" % (job_cpus))

        target_host = get_free_host(job_cpus)
        jobid = self.job_counter
        self.job_counter += 1

        if self.log_prefix is None:
            prefix = ""
        else:
            prefix = self.log_prefix(jobid)

        pid = nohup_return_pid(cmd, target_host, log_prefix=prefix, sudo=sudo, **kwargs)
        self.jobs_by_host[target_host][pid] = (jobid, job_cpus)
        self.hosts_by_job[jobid] = (target_host, pid)
    
        return jobid

    def _delete_job(self, jobid):
        host, pid = self.hosts_by_job[jobid]
        del self.hosts_by_job[jobid]
        del self.jobs_by_host[host][pid]
        
    def kill_job(self, jobid):
        host, pid = self.hosts_by_job[jobid]
        run_as_host("kill %d" % pid, host, sudo=True)
        self._delete_job(jobid)

    def sync_jobs(self):
        
        def running_processes(host):
            pass

        jobs_to_delete = []
        for h, jobs in self.jobs_by_host.items():
            pids = running_processes(host)
            for pid, (jobid, job_cpus) in jobs.items():
                if pid not in running_processes:
                    jobs_to_delete.append(jobid)
        
        for jobid in jobs_to_delete:
            self._delete_job(jobid)
            
