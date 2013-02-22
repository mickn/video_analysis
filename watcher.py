#!/usr/bin/env python

'''watcher vdir
watches for vdir *analyze_antfarm* jobs in SSUSP.
Kills all, then kills 'summarize*' jobs, then launches lsf_analyze_antfarm.py <vdir>'''

import LSF, os, sys, re, time

def reset_run(vdir,alljstats=None):
    if alljstats is None:
        alljstats = [LSF.lsf_get_job_details(j) for j in LSF.lsf_jobs_dict().keys()]
    
    to_kill = [j['Job'] for j in alljstats if 'analyze_antfarm' in j.get('Command','') and vdir in j.get('Command','')] + [j['Job'] for j in alljstats if 'summarize' in j.get('Command','') and vdir in j.get('Output File','')]
    LSF.lsf_kill_jobs(to_kill)

    os.system('lsf_analyze_antfarm.py %s .' % vdir)
        
if __name__ == '__main__':
    vdir = sys.argv[1]
    
    alljstats = [LSF.lsf_get_job_details(j) for j in LSF.lsf_jobs_dict().keys()]
    analyze_stats = [j['Status'] for j in alljstats if 'analyze_antfarm' in j.get('Command','') and vdir in j.get('Command','')]
    while len(analyze_stats) > 0:
        if 'SSUSP' in analyze_stats:
            reset_run(vdir,alljstats)
        
        time.sleep(60)
        alljstats = [LSF.lsf_get_job_details(j) for j in LSF.lsf_jobs_dict().keys()]
        analyze_stats = [j['Status'] for j in alljstats if 'analyze_antfarm' in j.get('Command','') and vdir in j.get('Command','')]
