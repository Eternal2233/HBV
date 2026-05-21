
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import os

# ============================================================
# Calibri 字体强制加载（修复 Spyder/conda 环境找不到字体的问题）
# ============================================================
_CALIBRI_FONT_DIR = os.path.expanduser("~/Library/Fonts")
_CALIBRI_FONT_FILE = os.path.join(_CALIBRI_FONT_DIR, "calibri.ttf")

# 方法1：直接从路径加载（最可靠）
if os.path.isfile(_CALIBRI_FONT_FILE):
    _fm.fontManager.addfont(_CALIBRI_FONT_FILE)   # 注册到 matplotlib
    _FONT_FAMILY = _fm.FontProperties(fname=_CALIBRI_FONT_FILE).get_name()
    print(f"[Font] Loaded Calibri from: {_CALIBRI_FONT_FILE} -> name='{_FONT_FAMILY}'")
else:
    # 方法2：回退到 findSystemFonts 搜索
    _calibri_paths = [p for p in _fm.findSystemFonts() if "calibri" in p.lower()]
    _carlito_paths = [p for p in _fm.findSystemFonts() if "carlito" in p.lower()]
    if _calibri_paths:
        _FONT_FAMILY = _fm.FontProperties(fname=_calibri_paths[0]).get_name()
        print(f"[Font] Found Calibri via findSystemFonts: {_calibri_paths[0]}")
    elif _carlito_paths:
        _FONT_FAMILY = _fm.FontProperties(fname=_carlito_paths[0]).get_name()
        print(f"[Font] Falling back to Carlito: {_carlito_paths[0]}")
    else:
        _FONT_FAMILY = "DejaVu Sans"
        print(f"[Font] WARNING: No Calibri/Carlito found, using '{_FONT_FAMILY}'")

# 全局设置字体
plt.rcParams['font.family'] = _FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False

wd="/Users/eternal/Desktop/HBV-global/"
os.chdir(wd)

import atomica as at
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def safe_min(seq, default=-1):
    seq = list(seq)
    return min(seq) if seq else default
#%% Import model framework
F=at.ProjectFramework("hbv_gamma.xlsx") #updated to include maternal antivirals
runs=200 #number of model simulations

#%% Function to run model (with probabilistic uncertainity analysis)
def model_results(F, data, calib, res_name, runs):
    
    """ Runs the model and simulations, returns central and sampled data sets for analysis"""
    
    import atomica as at
    import numpy as np
    
    ## Run the model 
    D=at.ProjectData.from_spreadsheet("databooks/"+data, framework=F) 
    P=at.Project(framework=F, databook="databooks/"+data, sim_start=1990, sim_end=2100, sim_dt=0.25, do_run=False) 
    cal = P.make_parset()
    cal.load_calibration("calibrations/" + calib)  
    
    # Central Estimate (can be used if needed, but median of ~100 runs converges very well [i.e., no meaningful difference])
    res=P.run_sim(parset=cal, result_name = res_name) # can also expand code here to check calibrations if needed
    
    #Probabilistic sensitivity analysis
    np.random.seed(25012023)
    psa=P.parsets[0]
    psa_sample=psa.sample()
    psa_res=P.run_sampled_sims(cal, n_samples=runs)
    
    #Initiate dictionary for required results 
    central_est={}
    store_runs={}
    out_res=[{"pop":["alive", "total", "sum"]}, #total population size
             {"pop_u5":["alive", {"Under 5":["0-4M", "0-4F"]}, "sum"]}, #total under 5 population size
             {"prev":["prev", "total", "weighted"]}, #total population prevalence
             {"chb_pop":["chb_pop", "total", "weighted"]}, #total population living with CHB
             {"prev_u5":["prev", {"Under 5":["0-4M", "0-4F"]}, "weighted"]}, #under 5y prevalence
             {"mort":[":dd_hbv", "total", "sum"]}, #Total HBV mortality
             {"hcc_inc": ["flw_hcc", "total", "sum"]}, #HCC incidence
             {"chb_inc": ["tot_inc", "total", "sum"]}, #CHB incidence
             {"hbe_preg": ["eag_ott", "15-49F", "sum"]}, #HBeAg prevalence in pregnant women
             {"yld": ["yld", "total", "sum"]}, #years lost to disability
             {"births": ["b_rate",{"Under 5":["0-4M", "0-4F"]}, "sum" ]}, #total births
             {"bd_cov": ["bd", {"Under 5":["0-4M", "0-4F"]}, "weighted"]}, #birth dose coverage
             {"hb3_cov": ["hb3", {"Under 5":["0-4M", "0-4F"]}, "weighted"]}, #hb3 coverage
             {"dx_rate":["tot_dx", "total", "sum"]}, #annual (incident) diagnoses
             {"tx_cov": ["treat", "total", "sum"]}, #total treatment coverage
             {"dm_dx": [[{"dm_dx":"it_dx+icl_dx+ict_dx+ie_dx+cc_dx+dc_dx+hcc_dx"}], "total", "sum"]}, #disease management, no treatment (for costs)
             {"dm_tx":[[{"dm_tx":"icl_tx+ict_tx+ie_tx+cc_tx+dc_tx+hcc_tx"}], "total", "sum"]}, #disease management, treatment (for costs)
             {"pop_hcc": [[{"pop_hcc":"cc_dx+cc_tx+dc_dx+dc_tx"}], "total", "sum"]}, #population HCC surveillance (for costs)
             {"tgt_hcc": [[{"tgt_hcc":"it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"50+":["50-69M", "50-69F", "70+M", "70+F"]}, "sum"]}, #targeted HCC surveillance (for costs)
             {"tgt_hcc_b": [[{"tgt_hcc":"it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"40+":["15-49M", "15-49F"]}, "sum"]}, #targeted HCC surveillance (for costs (40-49))
             {"hsp_tx": [[{"hsp_tx":"dc_tx+hcc_tx"}], "total", "sum"]}, #hospitalizations among treated (for costs)
             {"hsp_utx":[[{"hsp_utx":"dc+dc_dx+hcc+hcc_dx"}], "total", "sum"]},#hospitalization among untreated (for costs)
             {"dx_prop": ["diag_cov", "total", "weighted"]}, #diagnosis coverage
             {"tx_prop": ["treat_cov", "total", "weighted"]}, #treatment coverage (among eligible) 
             {"mav_n": ["mav_births",{"Under 5":["0-4M", "0-4F"]}, "sum" ]}, #number of births recieving mAVs and HBIG,
             {"prg_scr": ["preg_scr_num", "15-49F", "sum"]}, #number of pregnant women screened for HBsAg,
             {"prg_hrs":["preg_scr_num", "15-49F", "sum"]}, #number of pregnant women screened for HBeAg
             {"fc_cov_val": ["fc_cov", "total", "weighted"]}] #FC coverage among treated population
    
    for i in out_res:
        for key, val in i.items():
            df=at.PlotData(res, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1).series[0].vals
            central_est[key]=df
    
    
    # Loop to get all outputs via ensemble method
    for i in out_res:
        for key,val in i.items():
            mapping_function = lambda x: at.PlotData(x,outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1)
            ensemble = at.Ensemble(mapping_function=mapping_function)
            ensemble.update(psa_res)
            df=pd.DataFrame([d.series[0].vals for d in ensemble.samples])
            store_runs[key]=np.array(df).T
            
    return store_runs, central_est

#%% Run model and get results for each WHO region (Global outcomes calculated post-hoc)

#Note: Estimated time per simulation (2-7 seconds, depending on computer specifications)

afr_bl_runs, afr_bl_cent=model_results(F, "AFR.xlsx", "AFR_calib.xlsx", "Status Quo", runs)
afr_s1_runs, afr_s1_cent=model_results(F, "AFR_S1.xlsx", "AFR_calib.xlsx", "FC Slow (15y)", runs)
afr_s2_runs, afr_s2_cent=model_results(F, "AFR_S2.xlsx", "AFR_calib.xlsx", "FC Mid (10y)", runs)
afr_s3_runs, afr_s3_cent=model_results(F, "AFR_S3.xlsx", "AFR_calib.xlsx", "FC Fast (5y)", runs)

amr_bl_runs, amr_bl_cent=model_results(F, "AMR.xlsx", "AMR_calib.xlsx", "Status Quo", runs)
amr_s1_runs, amr_s1_cent=model_results(F, "AMR_S1.xlsx", "AMR_calib.xlsx", "FC Slow (15y)", runs)
amr_s2_runs, amr_s2_cent=model_results(F, "AMR_S2.xlsx", "AMR_calib.xlsx", "FC Mid (10y)", runs)
amr_s3_runs, amr_s3_cent=model_results(F, "AMR_S3.xlsx", "AMR_calib.xlsx", "FC Fast (5y)", runs)

emr_bl_runs, emr_bl_cent=model_results(F, "EMR.xlsx", "EMR_calib.xlsx", "Status Quo", runs)
emr_s1_runs, emr_s1_cent=model_results(F, "EMR_S1.xlsx", "EMR_calib.xlsx", "FC Slow (15y)", runs)
emr_s2_runs, emr_s2_cent=model_results(F, "EMR_S2.xlsx", "EMR_calib.xlsx", "FC Mid (10y)", runs)
emr_s3_runs, emr_s3_cent=model_results(F, "EMR_S3.xlsx", "EMR_calib.xlsx", "FC Fast (5y)", runs)

eur_bl_runs, eur_bl_cent=model_results(F, "EUR.xlsx", "EUR_calib.xlsx", "Status Quo", runs)
eur_s1_runs, eur_s1_cent=model_results(F, "EUR_S1.xlsx", "EUR_calib.xlsx", "FC Slow (15y)", runs)
eur_s2_runs, eur_s2_cent=model_results(F, "EUR_S2.xlsx", "EUR_calib.xlsx", "FC Mid (10y)", runs)
eur_s3_runs, eur_s3_cent=model_results(F, "EUR_S3.xlsx", "EUR_calib.xlsx", "FC Fast (5y)", runs)

sear_bl_runs, sear_bl_cent=model_results(F, "SEAR.xlsx", "SEAR_calib.xlsx", "Status Quo", runs)
sear_s1_runs, sear_s1_cent=model_results(F, "SEAR_S1.xlsx", "SEAR_calib.xlsx", "FC Slow (15y)", runs)
sear_s2_runs, sear_s2_cent=model_results(F, "SEAR_S2.xlsx", "SEAR_calib.xlsx", "FC Mid (10y)", runs)
sear_s3_runs, sear_s3_cent=model_results(F, "SEAR_S3.xlsx", "SEAR_calib.xlsx", "FC Fast (5y)", runs)

wpr_bl_runs, wpr_bl_cent=model_results(F, "WPR.xlsx", "WPR_calib.xlsx", "Status Quo", runs)
wpr_s1_runs, wpr_s1_cent=model_results(F, "WPR_S1.xlsx", "WPR_calib.xlsx", "FC Slow (15y)", runs)
wpr_s2_runs, wpr_s2_cent=model_results(F, "WPR_S2.xlsx", "WPR_calib.xlsx", "FC Mid (10y)", runs)
wpr_s3_runs, wpr_s3_cent=model_results(F, "WPR_S3.xlsx", "WPR_calib.xlsx", "FC Fast (5y)", runs)

#%% Economic Outcome Calculations
def econ_analysis (cent, cent_s1, cent_s2, cent_s3, res, res_s1, res_s2, res_s3, reg, wd, cost_data, runs, cdisc_rate, hdisc_rate, fc_cost=5000):
        
    ## FIX: defensively sanitize all Atomica-derived parameters used in cost/DALY calcs.
    ## Atomica outputs NaN for compartments that don't exist yet in early sim years (e.g.
    ## dm_tx in 1990 when no one is on treatment). NaN then propagates into cum_tc -> NEB.
    _sanitize_keys = ("mort", "yld", "dm_tx", "dm_dx", "fc_cov_val", "tx_cov", "dx_rate",
                      "dx_prop", "pop", "births", "bd_cov", "hb3_cov", "mav_n", "prg_scr",
                      "prg_hrs", "hsp_utx", "hsp_tx", "pop_hcc", "tgt_hcc", "tgt_hcc_b",
                      "chb_inc", "hcc_inc", "prev", "prev_u5")
    for d in (cent, cent_s1, cent_s2, cent_s3,
              res, res_s1, res_s2, res_s3):
        for k in _sanitize_keys:
            if k in d:
                d[k] = np.nan_to_num(d[k], nan=0.0, posinf=0.0, neginf=0.0)
    
    #Discounting Array
    if cdisc_rate >1:
        cdisc_rate=cdisc_rate/100
    else:
        cdisc_rate=cdisc_rate
        
    if hdisc_rate >1:
        hdisc_rate=hdisc_rate/100
    else:
        hdisc_rate=hdisc_rate
        
    discount=np.zeros((len(np.arange(1990,2100,1)),4))
    discount[:,0]=np.arange(1990,2100,1)
    
    for idx,val in enumerate(discount[:,0]):
        if val <= 2025:
            discount[idx,1]=1 #consumption discount
            discount[idx,2]=0 
            discount[idx,3]=1 #health discount
        else:
            discount[idx,1:3]=(1+cdisc_rate)**(2025-val)
            discount[idx,3]=(1+hdisc_rate)**(2025-val)
    
    
    
    # Vaccination Costing (seems to be working fine, needs more than 30 runs, and extraction of total births needs checking)
    vax_costs=pd.read_excel("cost and agg data/costs.xlsx", sheet_name="vax")
    
    np.random.seed(25012023)
    
    bd_vax=vax_costs[reg].iloc[0]
    hb3_vax=vax_costs[reg].iloc[1]
    
    bd_vax_samp=np.random.triangular(vax_costs[reg].iloc[6],vax_costs[reg].iloc[0],vax_costs[reg].iloc[7], runs)
    hb3_vax_samp=np.random.triangular(vax_costs[reg].iloc[8],vax_costs[reg].iloc[1],vax_costs[reg].iloc[9 ], runs)
    
    cbd_cost_bl, chb3_cost_bl, cbd_cost_s1, chb3_cost_s1,cbd_cost_s2, chb3_cost_s2,cbd_cost_s3, chb3_cost_s3=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))   
    bd_cost_bl,hb3_cost_bl,bd_cost_s1,hb3_cost_s1,bd_cost_s2,hb3_cost_s2,bd_cost_s3,hb3_cost_s3=np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110, runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs))
    
    cbd_cost_bl[:,0]=cent["bd_cov"][:]*cent["births"][:]*bd_vax*discount[:,1]
    chb3_cost_bl[:,0]=cent["hb3_cov"][:]*cent["births"][:]*hb3_vax*discount[:,1]
    cbd_cost_s1[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_vax*discount[:,1]
    chb3_cost_s1[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_vax*discount[:,1]
    cbd_cost_s2[:,0]=cent_s2["bd_cov"][:]*cent_s2["births"][:]*bd_vax*discount[:,1]
    chb3_cost_s2[:,0]=cent_s2["hb3_cov"][:]*cent_s2["births"][:]*hb3_vax*discount[:,1]
    cbd_cost_s3[:,0]=cent_s3["bd_cov"][:]*cent_s3["births"][:]*bd_vax*discount[:,1]
    chb3_cost_s3[:,0]=cent_s3["hb3_cov"][:]*cent_s3["births"][:]*hb3_vax*discount[:,1]
   
   
    for run in range(runs):
        bd_cost_bl[:,run]=res["bd_cov"][:,run]*res["births"][:,run]*bd_vax_samp[run]*discount[:,1]
        hb3_cost_bl[:,run]=res["hb3_cov"][:,run]*res["births"][:,run]*hb3_vax_samp[run]*discount[:,1]
        bd_cost_s1[:,run]=res_s1["bd_cov"][:,run]*res_s1["births"][:,run]*bd_vax_samp[run]*discount[:,1]
        hb3_cost_s1[:,run]=res_s1["hb3_cov"][:,run]*res_s1["births"][:,run]*hb3_vax_samp[run]*discount[:,1]
        bd_cost_s2[:,run]=res_s2["bd_cov"][:,run]*res_s2["births"][:,run]*bd_vax_samp[run]*discount[:,1]
        hb3_cost_s2[:,run]=res_s2["hb3_cov"][:,run]*res_s2["births"][:,run]*hb3_vax_samp[run]*discount[:,1]
        bd_cost_s3[:,run]=res_s3["bd_cov"][:,run]*res_s3["births"][:,run]*bd_vax_samp[run]*discount[:,1]
        hb3_cost_s3[:,run]=res_s3["hb3_cov"][:,run]*res_s3["births"][:,run]*hb3_vax_samp[run]*discount[:,1]
    
    care_costs=pd.read_excel("cost and agg data/costs.xlsx", sheet_name="care")
    dx_cost=care_costs[reg].iloc[0]
    hrs_cost=vax_costs[reg].iloc[10]
    mav_cost=vax_costs[reg].iloc[11]
    
    #msc- mAV and screening costs (plus HBIG)
    cmsc_cost_bl, cmsc_cost_s1,cmsc_cost_s2,cmsc_cost_s3=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    msc_cost_bl, msc_cost_s1,msc_cost_s2,msc_cost_s3=np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110, runs)),np.zeros((110,runs))
    
    cmsc_cost_bl[:,0]=(cent["prg_scr"][:]*dx_cost*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    cmsc_cost_s1[:,0]=(cent_s1["mav_n"][:]*mav_cost*discount[:,1])+(cent_s1["prg_scr"][:]*dx_cost*discount[:,1])+(cent_s1["prg_hrs"][:]*hrs_cost*discount[:,2])
    cmsc_cost_s2[:,0]=(cent_s2["mav_n"][:]*mav_cost*discount[:,1])+(cent_s2["prg_scr"][:]*dx_cost*discount[:,1])+(cent_s2["prg_hrs"][:]*hrs_cost*discount[:,2])
    cmsc_cost_s3[:,0]=(cent_s3["mav_n"][:]*mav_cost*discount[:,1])+(cent_s3["prg_scr"][:]*dx_cost*discount[:,1])+(cent_s3["prg_hrs"][:]*hrs_cost*discount[:,2])
    
    for run in range(runs):
        msc_cost_bl[:,run]=(res["prg_scr"][:,run]*dx_cost*discount[:,1])#(res["mav_n"][:,run]*mav_cost*discount[:,1])+(res["prg_hrs"][:,run]*hrs_cost*discount[:,1])
        msc_cost_s1[:,run]=(res_s1["mav_n"][:,run]*mav_cost*discount[:,1])+(res_s1["prg_scr"][:,run]*dx_cost*discount[:,1])+(res_s1["prg_hrs"][:,run]*hrs_cost*discount[:,2])
        msc_cost_s2[:,run]=(res_s2["mav_n"][:,run]*mav_cost*discount[:,1])+(res_s2["prg_scr"][:,run]*dx_cost*discount[:,1])+(res_s2["prg_hrs"][:,run]*hrs_cost*discount[:,2])
        msc_cost_s3[:,run]=(res_s3["mav_n"][:,run]*mav_cost*discount[:,1])+(res_s3["prg_scr"][:,run]*dx_cost*discount[:,1])+(res_s3["prg_hrs"][:,run]*hrs_cost*discount[:,2])

    # Diagnosis Costs (note three approaches coded, final one [dx_costb] used for analysis [proportional negative testing costs])
    cdx_cost_bl, cdx_cost_s1, cdx_cost_s2, cdx_cost_s3 = np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    dx_cost_bl, dx_cost_s1, dx_cost_s2, dx_cost_s3 = np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs))
    
    cdx_cost_bl[:,0]=cent["dx_rate"][:]*dx_cost*discount[:,1]
    cdx_cost_s1[:,0]=cent_s1["dx_rate"][:]*dx_cost*discount[:,1]
    cdx_cost_s2[:,0]=cent_s2["dx_rate"][:]*dx_cost*discount[:,1]
    cdx_cost_s3[:,0]=cent_s3["dx_rate"][:]*dx_cost*discount[:,1]
    
    for run in range(runs):
        dx_cost_bl[:,run]=res["dx_rate"][:,run]*dx_cost*discount[:,1]
        dx_cost_s1[:,run]=res_s1["dx_rate"][:, run]*dx_cost*discount[:,1]
        dx_cost_s2[:,run]=res_s2["dx_rate"][:, run]*dx_cost*discount[:,1]
        dx_cost_s3[:,run]=res_s3["dx_rate"][:, run]*dx_cost*discount[:,1]
        
    
    ## Dx Cost_alt1 --> 1:5 ratio pos to neg
    cdx_costa_bl, cdx_costa_s1, cdx_costa_s2, cdx_costa_s3 = np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    dx_costa_bl, dx_costa_s1, dx_costa_s2, dx_costa_s3 = np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs))
    
    cdx_costa_bl[:,0]=cent["dx_rate"][:]*(dx_cost*5)*discount[:,1]
    cdx_costa_s1[:,0]=cent_s1["dx_rate"][:]*(dx_cost*5)*discount[:,1]
    cdx_costa_s2[:,0]=cent_s2["dx_rate"][:]*(dx_cost*5)*discount[:,1]
    cdx_costa_s3[:,0]=cent_s3["dx_rate"][:]*(dx_cost*5)*discount[:,1]
    
    for run in range(runs):
        dx_costa_bl[:,run]=res["dx_rate"][:,run]*(dx_cost*5)*discount[:,1]
        dx_costa_s1[:,run]=res_s1["dx_rate"][:, run]*(dx_cost*5)*discount[:,1]
        dx_costa_s2[:,run]=res_s2["dx_rate"][:, run]*(dx_cost*5)*discount[:,1]
        dx_costa_s3[:,run]=res_s3["dx_rate"][:, run]*(dx_cost*5)*discount[:,1]
    
    ## Dx Cost_alt2 --> Proportional negative tests
    cbl_dx_inc, cs1_dx_inc,cs2_dx_inc,cs3_dx_inc=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    bl_dx_inc, s1_dx_inc,s2_dx_inc,s3_dx_inc=np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs))
    
    for i in range(len(res["dx_prop"])):
        if i < 1:
            cbl_dx_inc[i,0]=cent["dx_prop"][i]
            cs1_dx_inc[i,0]=cent_s1["dx_prop"][i]
            cs2_dx_inc[i,0]=cent_s2["dx_prop"][i]
            cs3_dx_inc[i,0]=cent_s3["dx_prop"][i]
        else:
            cbl_dx_inc[i,0]=max(cent["dx_prop"][i]-cent["dx_prop"][i-1],0)
            cs1_dx_inc[i,0]=max(cent_s1["dx_prop"][i]-cent_s1["dx_prop"][i-1],0)
            cs2_dx_inc[i,0]=max(cent_s2["dx_prop"][i]-cent_s2["dx_prop"][i-1],0)
            cs3_dx_inc[i,0]=max(cent_s3["dx_prop"][i]-cent_s3["dx_prop"][i-1],0)
    
    for i in range(len(res["dx_prop"])):
        for run in range(runs):
            if i < 1:
                bl_dx_inc[i,run]=res["dx_prop"][i,run]
                s1_dx_inc[i,run]=res_s1["dx_prop"][i,run]
                s2_dx_inc[i,run]=res_s2["dx_prop"][i,run]
                s3_dx_inc[i,run]=res_s3["dx_prop"][i,run]
            else:
                bl_dx_inc[i,run]=max(res["dx_prop"][i,run]-res["dx_prop"][i-1,run],0)
                s1_dx_inc[i,run]=max(res_s1["dx_prop"][i,run]-res_s1["dx_prop"][i-1,run],0)
                s2_dx_inc[i,run]=max(res_s2["dx_prop"][i,run]-res_s2["dx_prop"][i-1,run],0)
                s3_dx_inc[i,run]=max(res_s3["dx_prop"][i,run]-res_s3["dx_prop"][i-1,run],0)
        
    cdx_costb_bl, cdx_costb_s1, cdx_costb_s2, cdx_costb_s3 = np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    dx_costb_bl, dx_costb_s1, dx_costb_s2, dx_costb_s3 = np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs))
    
    for yr in range(len(cdx_costb_bl)):
        cdx_costb_bl[yr,0]=(cent["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*cbl_dx_inc[yr,0]*(cent["pop"][yr]-cent["dx_rate"][yr])*discount[yr,1])
        cdx_costb_s1[yr,0]=(cent_s1["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*cs1_dx_inc[yr,0]*(cent_s1["pop"][yr]-cent_s1["dx_rate"][yr])*discount[yr,1])
        cdx_costb_s2[yr,0]=(cent_s2["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*cs2_dx_inc[yr,0]*(cent_s2["pop"][yr]-cent_s2["dx_rate"][yr])*discount[yr,1])
        cdx_costb_s3[yr,0]=(cent_s3["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*cs3_dx_inc[yr,0]*(cent_s3["pop"][yr]-cent_s3["dx_rate"][yr])*discount[yr,1])
    
    for run in range(runs):
        for yr in range(len(dx_costb_bl)):
            dx_costb_bl[yr,run]=(res["dx_rate"][yr,run]*dx_cost*discount[yr,1])+(dx_cost*bl_dx_inc[yr,run]*(res["pop"][yr,run]-res["dx_rate"][yr,run])*discount[yr,1])
            dx_costb_s1[yr,run]=(res_s1["dx_rate"][yr, run]*dx_cost*discount[yr,1])+(dx_cost*s1_dx_inc[yr,run]*(res_s1["pop"][yr,run]-res_s1["dx_rate"][yr,run])*discount[yr,1])
            dx_costb_s2[yr,run]=(res_s2["dx_rate"][yr, run]*dx_cost*discount[yr,1])+(dx_cost*s2_dx_inc[yr,run]*(res_s2["pop"][yr,run]-res_s2["dx_rate"][yr,run])*discount[yr,1])
            dx_costb_s3[yr,run]=(res_s3["dx_rate"][yr, run]*dx_cost*discount[yr,1])+(dx_cost*s3_dx_inc[yr,run]*(res_s3["pop"][yr,run]-res_s3["dx_rate"][yr,run])*discount[yr,1])

    # Treatment Costs
    tx_cost=care_costs[reg].iloc[3]
    
    ctx_cost_bl, ctx_cost_s1, ctx_cost_s2, ctx_cost_s3 = np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    tx_cost_bl, tx_cost_s1, tx_cost_s2, tx_cost_s3 = np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs)),np.zeros((110,runs))
    
    ctx_cost_bl[:,0]=cent["tx_cov"][:]*tx_cost*discount[:,1]
    ctx_cost_s1[:,0]=cent_s1["tx_cov"][:]*tx_cost*discount[:,1]
    ctx_cost_s2[:,0]=cent_s2["tx_cov"][:]*tx_cost*discount[:,1]
    ctx_cost_s3[:,0]=cent_s3["tx_cov"][:]*tx_cost*discount[:,1]
    
    for run in range(runs):
        tx_cost_bl[:,run]=res["tx_cov"][:,run]*tx_cost*discount[:,1]
        tx_cost_s1[:, run]=res_s1["tx_cov"][:, run]*tx_cost*discount[:,1]
        tx_cost_s2[:, run]=res_s2["tx_cov"][:, run]*tx_cost*discount[:,1]
        tx_cost_s3[:, run]=res_s3["tx_cov"][:, run]*tx_cost*discount[:,1]
    

    ## NEW: Functional Cure Drug Costs ($5000/person/year, sensitivity $2000/$10000)
    ## FC cost = treated patients * annual NEW FC recipients (coverage increment) * drug cost per person
    ## FC is a one-time cure: only newly covered patients each year incur the drug cost.
    ## Using the annual increment (diff) of fc_cov_val avoids re-charging already-cured patients.
    ## Baseline has fc_cov=0 so no FC expenditure; scenarios have fc_cov>0
    fc_drug_cost = fc_cost  # passed as parameter, default $5000

    cfc_cost_bl, cfc_cost_s1 = np.zeros((110, 1)), np.zeros((110, 1))
    cfc_cost_s2, cfc_cost_s3 = np.zeros((110, 1)), np.zeros((110, 1))
    fc_psa_bl, fc_psa_s1 = np.zeros((110, runs)), np.zeros((110, runs))
    fc_psa_s2, fc_psa_s3 = np.zeros((110, runs)), np.zeros((110, runs))

    # PSA sampling for FC drug price (triangular: lb=$2000, mode=$5000, ub=$10000)
    fc_drug_samp = np.random.triangular(2000, fc_drug_cost, 10000, runs)

    # Annual increment of FC coverage (new recipients only); clip negatives to 0
    fc_diff_cbl = np.clip(np.diff(cent["fc_cov_val"][:],    prepend=0), 0, None)
    fc_diff_cs1 = np.clip(np.diff(cent_s1["fc_cov_val"][:], prepend=0), 0, None)
    fc_diff_cs2 = np.clip(np.diff(cent_s2["fc_cov_val"][:], prepend=0), 0, None)
    fc_diff_cs3 = np.clip(np.diff(cent_s3["fc_cov_val"][:], prepend=0), 0, None)

    # Central: FC cost = treated patients * coverage increment * drug cost * discount
    # Baseline: fc_cov=0 in baseline databook, so cfc_cost_bl stays zero
    cfc_cost_bl[:, 0] = cent["dm_tx"][:]     * fc_diff_cbl * fc_drug_cost * discount[:, 1]
    cfc_cost_s1[:, 0] = cent_s1["dm_tx"][:] * fc_diff_cs1 * fc_drug_cost * discount[:, 1]
    cfc_cost_s2[:, 0] = cent_s2["dm_tx"][:] * fc_diff_cs2 * fc_drug_cost * discount[:, 1]
    cfc_cost_s3[:, 0] = cent_s3["dm_tx"][:] * fc_diff_cs3 * fc_drug_cost * discount[:, 1]

    # Sampled (with price uncertainty) — same increment logic per run
    for run in range(runs):
        fc_diff_bl_r = np.clip(np.diff(res["fc_cov_val"][:,    run], prepend=0), 0, None)
        fc_diff_s1_r = np.clip(np.diff(res_s1["fc_cov_val"][:, run], prepend=0), 0, None)
        fc_diff_s2_r = np.clip(np.diff(res_s2["fc_cov_val"][:, run], prepend=0), 0, None)
        fc_diff_s3_r = np.clip(np.diff(res_s3["fc_cov_val"][:, run], prepend=0), 0, None)
        fc_psa_bl[:, run] = res["dm_tx"][:,    run] * fc_diff_bl_r * fc_drug_samp[run] * discount[:, 1]
        fc_psa_s1[:, run] = res_s1["dm_tx"][:, run] * fc_diff_s1_r * fc_drug_samp[run] * discount[:, 1]
        fc_psa_s2[:, run] = res_s2["dm_tx"][:, run] * fc_diff_s2_r * fc_drug_samp[run] * discount[:, 1]
        fc_psa_s3[:, run] = res_s3["dm_tx"][:, run] * fc_diff_s3_r * fc_drug_samp[run] * discount[:, 1]
    
    ## Direct Medical Costs (already discounted; central) - includes FC drug cost
    dmc_cent_bl=cbd_cost_bl+chb3_cost_bl+cdx_costb_bl+ctx_cost_bl+cmsc_cost_bl+cfc_cost_bl
    dmc_cent_s1=cbd_cost_s1+chb3_cost_s1+cdx_costb_s1+ctx_cost_s1+cmsc_cost_s1+cfc_cost_s1
    dmc_cent_s2=cbd_cost_s2+chb3_cost_s2+cdx_costb_s2+ctx_cost_s2+cmsc_cost_s2+cfc_cost_s2
    dmc_cent_s3=cbd_cost_s3+chb3_cost_s3+cdx_costb_s3+ctx_cost_s3+cmsc_cost_s3+cfc_cost_s3
    
    ## Direct Medical Costs (already discounted; sampled)
    dmc_psa_bl=bd_cost_bl+hb3_cost_bl+dx_costb_bl+tx_cost_bl+msc_cost_bl+fc_psa_bl
    dmc_psa_s1=bd_cost_s1+hb3_cost_s1+dx_costb_s1+tx_cost_s1+msc_cost_s1+fc_psa_s1
    dmc_psa_s2=bd_cost_s2+hb3_cost_s2+dx_costb_s2+tx_cost_s2+msc_cost_s2+fc_psa_s2
    dmc_psa_s3=bd_cost_s3+hb3_cost_s3+dx_costb_s3+tx_cost_s3+msc_cost_s3+fc_psa_s3
    
    ## Disease Progression Management costs 
    util=0.25  #care utilization assumption
    tx_hosp=0.5  #treatment preventing hospitalisation assumption
    
    ## Indirect Disease Management Costs (assume 50% HR covered under UHC; 25% hospital utilization)
    dx_dmc=care_costs[reg].iloc[4] #cost of disease management (diagnosed)
    tx_dmc=care_costs[reg].iloc[5] #cost of disease management (on treatment)
    hosp_cost=care_costs[reg].iloc[7] #cost of hospitalisation
    hcc_cost=care_costs[reg].iloc[6] #HCC surveillance costs
    hcc_prp=care_costs[reg].iloc[8] #proportion of CHB in 15-49y who are 40-49y (HCC screening)
    
    cmc_cost_bl, cmc_cost_s1, cmc_cost_s2, cmc_cost_s3= np.zeros((110,1)),np.zeros((110, 1)),np.zeros((110, 1)),np.zeros((110, 1))
    chosp_cost_bl, chosp_cost_s1, chosp_cost_s2, chosp_cost_s3=np.zeros((110, 1)),np.zeros((110,1)),np.zeros((110, 1)),np.zeros((110, 1))
    chcc_cost_bl, chcc_cost_s1, chcc_cost_s2, chcc_cost_s3=np.zeros((110, 1)),np.zeros((110, 1)),np.zeros((110, 1)),np.zeros((110, 1))
    
    ## Disease Management Costs
    cmc_cost_bl[:, 0]=((cent["dm_dx"][:]*dx_dmc*util)+(cent["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    cmc_cost_s1[:, 0]=((cent_s1["dm_dx"][:]*dx_dmc*util)+(cent_s1["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    cmc_cost_s2[:, 0]=((cent_s2["dm_dx"][:]*dx_dmc*util)+(cent_s2["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    cmc_cost_s3[:, 0]=((cent_s3["dm_dx"][:]*dx_dmc*util)+(cent_s3["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    ## Hospitalisation Costs
    chosp_cost_bl[:, 0]=((cent["hsp_utx"][:]*hosp_cost*util)+(cent["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    chosp_cost_s1[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util)+(cent_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    chosp_cost_s2[:, 0]=((cent_s2["hsp_utx"][:]*hosp_cost*util)+(cent_s2["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    chosp_cost_s3[:, 0]=((cent_s3["hsp_utx"][:]*hosp_cost*util)+(cent_s3["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    ## HCC Surveillance Costs (assume 50% HR covered under UHC; 25% resource utilization)
    chcc_cost_bl[:, 0]=((cent["pop_hcc"][:]*hcc_cost*util)+(cent["tgt_hcc"][:]*hcc_cost*util)+(cent["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    chcc_cost_s1[:, 0]=((cent_s1["pop_hcc"][:]*hcc_cost*util)+(cent_s1["tgt_hcc"][:]*hcc_cost*util)+(cent_s1["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    chcc_cost_s2[:, 0]=((cent_s2["pop_hcc"][:]*hcc_cost*util)+(cent_s2["tgt_hcc"][:]*hcc_cost*util)+(cent_s2["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    chcc_cost_s3[:, 0]=((cent_s3["pop_hcc"][:]*hcc_cost*util)+(cent_s3["tgt_hcc"][:]*hcc_cost*util)+(cent_s3["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    
    
    mc_cost_bl, mc_cost_s1, mc_cost_s2, mc_cost_s3= np.zeros((110,runs)),np.zeros((110, runs)),np.zeros((110, runs)),np.zeros((110, runs))
    hosp_cost_bl, hosp_cost_s1, hosp_cost_s2, hosp_cost_s3=np.zeros((110, runs)),np.zeros((110, runs)),np.zeros((110, runs)),np.zeros((110, runs))
    hcc_cost_bl, hcc_cost_s1, hcc_cost_s2, hcc_cost_s3=np.zeros((110, runs)),np.zeros((110, runs)),np.zeros((110, runs)),np.zeros((110, runs))
    
    for run in range(runs):
        ## Disease Management Costs
        mc_cost_bl[:, run]=((res["dm_dx"][:, run]*dx_dmc*util)+(res["dm_tx"][:, run]*tx_dmc*util))*discount[:,1]
        mc_cost_s1[:, run]=((res_s1["dm_dx"][:, run]*dx_dmc*util)+(res_s1["dm_tx"][:, run]*tx_dmc*util))*discount[:,1]
        mc_cost_s2[:, run]=((res_s2["dm_dx"][:, run]*dx_dmc*util)+(res_s2["dm_tx"][:, run]*tx_dmc*util))*discount[:,1]
        mc_cost_s3[:, run]=((res_s3["dm_dx"][:, run]*dx_dmc*util)+(res_s3["dm_tx"][:, run]*tx_dmc*util))*discount[:,1]
        ## Hospitalisation Costs
        hosp_cost_bl[:, run]=((res["hsp_utx"][:,run]*hosp_cost*util)+(res["hsp_tx"][:, run]*hosp_cost*util*tx_hosp))*discount[:,1]
        hosp_cost_s1[:, run]=((res_s1["hsp_utx"][:,run]*hosp_cost*util)+(res_s1["hsp_tx"][:, run]*hosp_cost*util*tx_hosp))*discount[:,1]
        hosp_cost_s2[:, run]=((res_s2["hsp_utx"][:,run]*hosp_cost*util)+(res_s2["hsp_tx"][:, run]*hosp_cost*util*tx_hosp))*discount[:,1]
        hosp_cost_s3[:, run]=((res_s3["hsp_utx"][:,run]*hosp_cost*util)+(res_s3["hsp_tx"][:, run]*hosp_cost*util*tx_hosp))*discount[:,1]
        ## HCC Surveillance Costs (assume 50% HR covered under UHC; 25% resource utilization)
        hcc_cost_bl[:, run]=((res["pop_hcc"][:, run]*hcc_cost*util)+(res["tgt_hcc"][:, run]*hcc_cost*util)+(res["tgt_hcc_b"][:,run]*hcc_prp*hcc_cost*util))*discount[:,1]
        hcc_cost_s1[:, run]=((res_s1["pop_hcc"][:, run]*hcc_cost*util)+(res_s1["tgt_hcc"][:, run]*hcc_cost*util)+(res_s1["tgt_hcc_b"][:,run]*hcc_prp*hcc_cost*util))*discount[:,1]
        hcc_cost_s2[:, run]=((res_s2["pop_hcc"][:, run]*hcc_cost*util)+(res_s2["tgt_hcc"][:, run]*hcc_cost*util)+(res_s2["tgt_hcc_b"][:,run]*hcc_prp*hcc_cost*util))*discount[:,1]
        hcc_cost_s3[:, run]=((res_s3["pop_hcc"][:, run]*hcc_cost*util)+(res_s3["tgt_hcc"][:, run]*hcc_cost*util)+(res_s3["tgt_hcc_b"][:,run]*hcc_prp*hcc_cost*util))*discount[:,1]
     
    ## Indirect Medical Costs (discounted; central)
    imc_cent_bl=cmc_cost_bl+chosp_cost_bl+chcc_cost_bl
    imc_cent_s1=cmc_cost_s1+chosp_cost_s1+chcc_cost_s1
    imc_cent_s2=cmc_cost_s2+chosp_cost_s2+chcc_cost_s2
    imc_cent_s3=cmc_cost_s3+chosp_cost_s3+chcc_cost_s3

    ## Indirect Medical Costs (discounted; sampled)
    imc_psa_bl=mc_cost_bl+hosp_cost_bl+hcc_cost_bl
    imc_psa_s1=mc_cost_s1+hosp_cost_s1+hcc_cost_s1
    imc_psa_s2=mc_cost_s2+hosp_cost_s2+hcc_cost_s2
    imc_psa_s3=mc_cost_s3+hosp_cost_s3+hcc_cost_s3
    
    
    ## Productivity Loss (calculate for baseline, check outcomes, then repeat) 
    prod_costs=pd.read_excel(cost_data, sheet_name="emp_gdp_lex")
    etp_ratio=prod_costs[reg].iloc[0]
    gdp=prod_costs[reg].iloc[1]
    life_exp=prod_costs[reg].iloc[2]
    
    ## GDP growth
    gdp_grw=np.zeros((len(np.arange(1990,2100,1)),4))
    gdp_grw[:,0]=np.arange(1990,2100,1)
    gdp_grw[:,1:4]=gdp
    
    for i,val in enumerate(gdp_grw[:,0]):
        if val>2026:
            gdp_grw[i,1]=gdp_grw[i-1,1]*1.00
            gdp_grw[i,2]=gdp_grw[i-1,2]*1.015
            gdp_grw[i,3]=gdp_grw[i-1,3]*1.03
            
    age_of_deaths=np.array([0.01, 0.031, 0.253, 0.341, 0.365])
    prop_leaving_age_categories=np.array([1/15, 1/15, 1/20, 1/15])
    all_cause_mort=np.array([0.003, 0.0013, 0.0022, 0.0103, (1/life_exp)])
    
    ## Baseline (central)
    cbl_deaths=np.zeros((len(cent["mort"]),2))
    cbl_deaths[:,0]=np.arange(1990,2100,1)
    cbl_deaths[:,1]=cent["mort"]
    
    for idx,val in enumerate(cbl_deaths[:,0]):
        if val < 2026:
            cbl_deaths[idx,1]=0
    
    ghosts_cbl=np.zeros((len(cbl_deaths), len(age_of_deaths)))
    ghosts_cbl[0,:]=cbl_deaths[0,1]*age_of_deaths
    
    for t in range(1,len(cbl_deaths)):
        ppl_who_age=ghosts_cbl[t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
        ghosts_cbl[t,0]=max(0, ghosts_cbl[t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_cbl[t-1,0])
        ghosts_cbl[t,1]=max(0, ghosts_cbl[t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_cbl[t-1,1])
        ghosts_cbl[t,2]=max(0, ghosts_cbl[t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_cbl[t-1,2])
        ghosts_cbl[t,3]=max(0, ghosts_cbl[t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_cbl[t-1,3])
        ghosts_cbl[t,4]=max(0, ghosts_cbl[t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_cbl[t-1,4])
        
        ghosts_cbl[t,:]= ghosts_cbl[t,:]+cbl_deaths[t,1]*age_of_deaths
    
    ## Baseline (sampled)
    bl_deaths=np.zeros((len(res["mort"]), runs+1))
    bl_deaths[:,0]=np.arange(1990,2100,1)
    bl_deaths[:,1:]=res["mort"]
    
    for idx,val in enumerate(bl_deaths[:,0]):
        if val < 2026:
            bl_deaths[idx,1:]=0
    
    ghosts_bl=np.zeros((runs,len(bl_deaths),len(age_of_deaths))) 
    
    for run in range(runs):
        ghosts_bl[run,0,:]=bl_deaths[0,run+1]*age_of_deaths
    
    for run in range(runs):
        for t in range(1,len(bl_deaths)):
            ppl_who_age=ghosts_bl[run,t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
            ghosts_bl[run,t,0]=max(0, ghosts_bl[run,t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_bl[run,t-1,0])
            ghosts_bl[run,t,1]=max(0, ghosts_bl[run,t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_bl[run,t-1,1])
            ghosts_bl[run,t,2]=max(0, ghosts_bl[run,t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_bl[run,t-1,2])
            ghosts_bl[run,t,3]=max(0, ghosts_bl[run,t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_bl[run,t-1,3])
            ghosts_bl[run,t,4]=max(0, ghosts_bl[run,t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_bl[run,t-1,4])
        
            ghosts_bl[run,t,:]= ghosts_bl[run,t,:]+bl_deaths[t,run+1]*age_of_deaths

    ## Scenario 1 (central)
    cs1_deaths=np.zeros((len(cent_s1["mort"]),2))
    cs1_deaths[:,0]=np.arange(1990,2100,1)
    cs1_deaths[:,1]=cent_s1["mort"]
     
    for idx,val in enumerate(cs1_deaths[:,0]):
        if val < 2026:
            cs1_deaths[idx,1]=0
     
    ghosts_cs1=np.zeros((len(cs1_deaths), len(age_of_deaths)))
    ghosts_cs1[0,:]=cs1_deaths[0,1]*age_of_deaths
     
    for t in range(1,len(cs1_deaths)):
        ppl_who_age=ghosts_cs1[t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
        ghosts_cs1[t,0]=max(0, ghosts_cs1[t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_cs1[t-1,0])
        ghosts_cs1[t,1]=max(0, ghosts_cs1[t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_cs1[t-1,1])
        ghosts_cs1[t,2]=max(0, ghosts_cs1[t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_cs1[t-1,2])
        ghosts_cs1[t,3]=max(0, ghosts_cs1[t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_cs1[t-1,3])
        ghosts_cs1[t,4]=max(0, ghosts_cs1[t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_cs1[t-1,4])
         
        ghosts_cs1[t,:]= ghosts_cs1[t,:]+cs1_deaths[t,1]*age_of_deaths
        
        
    ## Scenario 1 (sampled)
    s1_deaths=np.zeros((len(res_s1["mort"]), runs+1))
    s1_deaths[:,0]=np.arange(1990,2100,1)
    s1_deaths[:,1:]=res_s1["mort"]
    
    for idx,val in enumerate(s1_deaths[:,0]):
        if val < 2026:
            s1_deaths[idx,1:]=0
    
    ghosts_s1=np.zeros((runs,len(s1_deaths),len(age_of_deaths))) 
    
    for run in range(runs):
        ghosts_s1[run,0,:]=s1_deaths[0,run+1]*age_of_deaths
    
    for run in range(runs):
        for t in range(1,len(s1_deaths)):
            ppl_who_age=ghosts_s1[run,t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
            ghosts_s1[run,t,0]=max(0, ghosts_s1[run,t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_s1[run,t-1,0])
            ghosts_s1[run,t,1]=max(0, ghosts_s1[run,t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_s1[run,t-1,1])
            ghosts_s1[run,t,2]=max(0, ghosts_s1[run,t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_s1[run,t-1,2])
            ghosts_s1[run,t,3]=max(0, ghosts_s1[run,t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_s1[run,t-1,3])
            ghosts_s1[run,t,4]=max(0, ghosts_s1[run,t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_s1[run,t-1,4])
        
            ghosts_s1[run,t,:]= ghosts_s1[run,t,:]+s1_deaths[t,run+1]*age_of_deaths
    
    ## Scenario 2 (central)
    cs2_deaths=np.zeros((len(cent_s2["mort"]),2))
    cs2_deaths[:,0]=np.arange(1990,2100,1)
    cs2_deaths[:,1]=cent_s2["mort"]
     
    for idx,val in enumerate(cs2_deaths[:,0]):
        if val < 2026:
            cs2_deaths[idx,1]=0
     
    ghosts_cs2=np.zeros((len(cs2_deaths), len(age_of_deaths)))
    ghosts_cs2[0,:]=cs2_deaths[0,1]*age_of_deaths
     
    for t in range(1,len(cs2_deaths)):
        ppl_who_age=ghosts_cs2[t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
        ghosts_cs2[t,0]=max(0, ghosts_cs2[t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_cs2[t-1,0])
        ghosts_cs2[t,1]=max(0, ghosts_cs2[t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_cs2[t-1,1])
        ghosts_cs2[t,2]=max(0, ghosts_cs2[t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_cs2[t-1,2])
        ghosts_cs2[t,3]=max(0, ghosts_cs2[t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_cs2[t-1,3])
        ghosts_cs2[t,4]=max(0, ghosts_cs2[t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_cs2[t-1,4])
         
        ghosts_cs2[t,:]= ghosts_cs2[t,:]+cs2_deaths[t,1]*age_of_deaths
        
    ## Scenario 2 (sampled)
    s2_deaths=np.zeros((len(res_s2["mort"]), runs+1))
    s2_deaths[:,0]=np.arange(1990,2100,1)
    s2_deaths[:,1:]=res_s2["mort"]
     
    for idx,val in enumerate(s2_deaths[:,0]):
        if val < 2026:
            s2_deaths[idx,1:]=0
     
    ghosts_s2=np.zeros((runs,len(s2_deaths),len(age_of_deaths))) 
     
    for run in range(runs):
        ghosts_s2[run,0,:]=s2_deaths[0,run+1]*age_of_deaths
     
    for run in range(runs):
        for t in range(1,len(s2_deaths)):
            ppl_who_age=ghosts_s2[run,t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
            ghosts_s2[run,t,0]=max(0, ghosts_s2[run,t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_s2[run,t-1,0])
            ghosts_s2[run,t,1]=max(0, ghosts_s2[run,t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_s2[run,t-1,1])
            ghosts_s2[run,t,2]=max(0, ghosts_s2[run,t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_s2[run,t-1,2])
            ghosts_s2[run,t,3]=max(0, ghosts_s2[run,t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_s2[run,t-1,3])
            ghosts_s2[run,t,4]=max(0, ghosts_s2[run,t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_s2[run,t-1,4])
         
            ghosts_s2[run,t,:]= ghosts_s2[run,t,:]+s2_deaths[t,run+1]*age_of_deaths
             
    ## Scenario 3 (central)
    cs3_deaths=np.zeros((len(cent_s3["mort"]),2))
    cs3_deaths[:,0]=np.arange(1990,2100,1)
    cs3_deaths[:,1]=cent_s3["mort"]
     
    for idx,val in enumerate(cs3_deaths[:,0]):
        if val < 2026:
            cs3_deaths[idx,1]=0
     
    ghosts_cs3=np.zeros((len(cs3_deaths), len(age_of_deaths)))
    ghosts_cs3[0,:]=cs3_deaths[0,1]*age_of_deaths
     
    for t in range(1,len(cs3_deaths)):
        ppl_who_age=ghosts_cs3[t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
        ghosts_cs3[t,0]=max(0, ghosts_cs3[t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_cs3[t-1,0])
        ghosts_cs3[t,1]=max(0, ghosts_cs3[t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_cs3[t-1,1])
        ghosts_cs3[t,2]=max(0, ghosts_cs3[t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_cs3[t-1,2])
        ghosts_cs3[t,3]=max(0, ghosts_cs3[t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_cs3[t-1,3])
        ghosts_cs3[t,4]=max(0, ghosts_cs3[t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_cs3[t-1,4])
         
        ghosts_cs3[t,:]= ghosts_cs3[t,:]+cs3_deaths[t,1]*age_of_deaths

    ## Scenario 3 (sampled)
    s3_deaths=np.zeros((len(res_s3["mort"]), runs+1))
    s3_deaths[:,0]=np.arange(1990,2100,1)
    s3_deaths[:,1:]=res_s3["mort"]
     
    for idx,val in enumerate(s3_deaths[:,0]):
        if val < 2026:
            s3_deaths[idx,1:]=0
     
    ghosts_s3=np.zeros((runs,len(s3_deaths),len(age_of_deaths))) 
     
    for run in range(runs):
        ghosts_s3[run,0,:]=s3_deaths[0,run+1]*age_of_deaths
     
    for run in range(runs):
        for t in range(1,len(s3_deaths)):
            ppl_who_age=ghosts_s3[run,t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
            ghosts_s3[run,t,0]=max(0, ghosts_s3[run,t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_s3[run,t-1,0])
            ghosts_s3[run,t,1]=max(0, ghosts_s3[run,t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_s3[run,t-1,1])
            ghosts_s3[run,t,2]=max(0, ghosts_s3[run,t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_s3[run,t-1,2])
            ghosts_s3[run,t,3]=max(0, ghosts_s3[run,t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_s3[run,t-1,3])
            ghosts_s3[run,t,4]=max(0, ghosts_s3[run,t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_s3[run,t-1,4])
         
            ghosts_s3[run,t,:]= ghosts_s3[run,t,:]+s3_deaths[t,run+1]*age_of_deaths
            

    ## DALYs 
    cbl_yll=np.sum(ghosts_cbl[:,0:5], axis=1)*discount[:,3]
    cs1_yll=np.sum(ghosts_cs1[:,0:5], axis=1)*discount[:,3]
    cs2_yll=np.sum(ghosts_cs2[:,0:5], axis=1)*discount[:,3]
    cs3_yll=np.sum(ghosts_cs3[:,0:5], axis=1)*discount[:,3]

    bl_yll=(np.sum(ghosts_bl[:,:,0:5], axis=2).T)
    s1_yll=(np.sum(ghosts_s1[:,:,0:5], axis=2).T)
    s2_yll=(np.sum(ghosts_s2[:,:,0:5], axis=2).T)
    s3_yll=(np.sum(ghosts_s3[:,:,0:5], axis=2).T)
   
    for idx in range(len(bl_yll)):  
        bl_yll[idx,:]=bl_yll[idx,:]*discount[idx,3]
        s1_yll[idx,:]=s1_yll[idx,:]*discount[idx,3]
        s2_yll[idx,:]=s2_yll[idx,:]*discount[idx,3]
        s3_yll[idx,:]=s3_yll[idx,:]*discount[idx,3]
    
    cbl_dalys=cbl_yll+cent["yld"]
    cs1_dalys=cs1_yll+cent_s1["yld"]
    cs2_dalys=cs2_yll+cent_s2["yld"]
    cs3_dalys=cs3_yll+cent_s3["yld"]
    
    bl_dalys=bl_yll+res["yld"]
    s1_dalys=s1_yll+res_s1["yld"]
    s2_dalys=s2_yll+res_s2["yld"]
    s3_dalys=s3_yll+res_s3["yld"]
    
    ## Productivity 
    cbl_prod=np.sum(ghosts_cbl[:,0:4], axis=1)*gdp_grw[:,2]*etp_ratio*discount[:,3]
    cs1_prod=np.sum(ghosts_cs1[:,0:4], axis=1)*gdp_grw[:,2]*etp_ratio*discount[:,3]
    cs2_prod=np.sum(ghosts_cs2[:,0:4], axis=1)*gdp_grw[:,2]*etp_ratio*discount[:,3]
    cs3_prod=np.sum(ghosts_cs3[:,0:4], axis=1)*gdp_grw[:,2]*etp_ratio*discount[:,3]

    bl_prod=(np.sum(ghosts_bl[:,:,0:4], axis=2).T)
    s1_prod=(np.sum(ghosts_s1[:,:,0:4], axis=2).T)
    s2_prod=(np.sum(ghosts_s2[:,:,0:4], axis=2).T)
    s3_prod=(np.sum(ghosts_s3[:,:,0:4], axis=2).T)
    
    for idx in range(len(bl_yll)):  
        bl_prod[idx,:]=bl_prod[idx,:]*gdp_grw[idx,2]*etp_ratio*discount[idx,3]
        s1_prod[idx,:]=s1_prod[idx,:]*gdp_grw[idx,2]*etp_ratio*discount[idx,3]
        s2_prod[idx,:]=s2_prod[idx,:]*gdp_grw[idx,2]*etp_ratio*discount[idx,3]
        s3_prod[idx,:]=s3_prod[idx,:]*gdp_grw[idx,2]*etp_ratio*discount[idx,3]
    
    
    ##(Note: Both Measures Require Cumulative Costs, only instantaneous calculated so far)
    ## Net Economic Benefit (up to here for central estimates)
    cbl_ins_tc=cbl_prod[:]+dmc_cent_bl[:,0]+imc_cent_bl[:,0]
    cs1_ins_tc=cs1_prod[:]+dmc_cent_s1[:,0]+imc_cent_s1[:,0]
    cs2_ins_tc=cs2_prod[:]+dmc_cent_s2[:,0]+imc_cent_s2[:,0]
    cs3_ins_tc=cs3_prod[:]+dmc_cent_s3[:,0]+imc_cent_s3[:,0]
    
    cbl_cum_tc=np.zeros(np.shape(cbl_ins_tc))
    cs1_cum_tc=np.zeros(np.shape(cs1_ins_tc))
    cs2_cum_tc=np.zeros(np.shape(cs2_ins_tc))
    cs3_cum_tc=np.zeros(np.shape(cs3_ins_tc))
    
    for i in range(len(cbl_ins_tc)):
        if i < 1:
            cbl_cum_tc[i]= cbl_ins_tc[i]
            cs1_cum_tc[i]= cs1_ins_tc[i]
            cs2_cum_tc[i]= cs2_ins_tc[i]
            cs3_cum_tc[i]= cs3_ins_tc[i]
        else:
            cbl_cum_tc[i]= cbl_cum_tc[i-1]+cbl_ins_tc[i]
            cs1_cum_tc[i]= cs1_cum_tc[i-1]+cs1_ins_tc[i]
            cs2_cum_tc[i]= cs2_cum_tc[i-1]+cs2_ins_tc[i]
            cs3_cum_tc[i]= cs3_cum_tc[i-1]+cs3_ins_tc[i]
    
    bl_ins_tc=bl_prod+dmc_psa_bl+imc_psa_bl
    s1_ins_tc=s1_prod+dmc_psa_s1+imc_psa_s1
    s2_ins_tc=s2_prod+dmc_psa_s2+imc_psa_s2
    s3_ins_tc=s3_prod+dmc_psa_s3+imc_psa_s3
    
    bl_cum_tc=np.zeros(np.shape(bl_ins_tc))
    s1_cum_tc=np.zeros(np.shape(s1_ins_tc))
    s2_cum_tc=np.zeros(np.shape(s2_ins_tc))
    s3_cum_tc=np.zeros(np.shape(s3_ins_tc))
    
    for i in range(len(bl_ins_tc)):
        if i < 1:
            bl_cum_tc[i,:]=bl_ins_tc[i,:]
            s1_cum_tc[i,:]=s1_ins_tc[i,:]
            s2_cum_tc[i,:]=s2_ins_tc[i,:]
            s3_cum_tc[i,:]=s3_ins_tc[i,:]
        else:
            bl_cum_tc[i,:]=bl_cum_tc[i-1,:]+bl_ins_tc[i,:]
            s1_cum_tc[i,:]=s1_cum_tc[i-1,:]+s1_ins_tc[i,:]
            s2_cum_tc[i,:]=s2_cum_tc[i-1,:]+s2_ins_tc[i,:]
            s3_cum_tc[i,:]=s3_cum_tc[i-1,:]+s3_ins_tc[i,:]
    
    cs1_neb=cbl_cum_tc-cs1_cum_tc
    cs2_neb=cbl_cum_tc-cs2_cum_tc
    cs3_neb=cbl_cum_tc-cs3_cum_tc
    
    s1_neb=bl_cum_tc-s1_cum_tc
    s2_neb=bl_cum_tc-s2_cum_tc
    s3_neb=bl_cum_tc-s3_cum_tc
    
    ## FIX: sanitize NEB arrays (NaN propagates from early-year Atomica params e.g. dm_tx in 1990)
    cs1_neb=np.nan_to_num(cs1_neb, nan=0.0, posinf=0.0, neginf=0.0)
    cs2_neb=np.nan_to_num(cs2_neb, nan=0.0, posinf=0.0, neginf=0.0)
    cs3_neb=np.nan_to_num(cs3_neb, nan=0.0, posinf=0.0, neginf=0.0)
    s1_neb=np.nan_to_num(s1_neb, nan=0.0, posinf=0.0, neginf=0.0)
    s2_neb=np.nan_to_num(s2_neb, nan=0.0, posinf=0.0, neginf=0.0)
    s3_neb=np.nan_to_num(s3_neb, nan=0.0, posinf=0.0, neginf=0.0)
   
    ## ICERs
    cbl_ins_dc=dmc_cent_bl[:,0]+imc_cent_bl[:,0]
    cs1_ins_dc=dmc_cent_s1[:,0]+imc_cent_s1[:,0]
    cs2_ins_dc=dmc_cent_s2[:,0]+imc_cent_s2[:,0]
    cs3_ins_dc=dmc_cent_s3[:,0]+imc_cent_s3[:,0]
    
    cbl_cum_dc=np.zeros(np.shape(cbl_ins_dc))
    cs1_cum_dc=np.zeros(np.shape(cs1_ins_dc))
    cs2_cum_dc=np.zeros(np.shape(cs2_ins_dc))
    cs3_cum_dc=np.zeros(np.shape(cs3_ins_dc))
   
    cbl_cum_daly=np.zeros(np.shape(cbl_dalys))
    cs1_cum_daly=np.zeros(np.shape(cs1_dalys))
    cs2_cum_daly=np.zeros(np.shape(cs2_dalys))
    cs3_cum_daly=np.zeros(np.shape(cs3_dalys))
    
    for i in range(len(cbl_ins_dc)):
        if i < 1:
            cbl_cum_dc[i]=cbl_ins_dc[i]
            cs1_cum_dc[i]=cs1_ins_dc[i]
            cs2_cum_dc[i]=cs2_ins_dc[i]
            cs3_cum_dc[i]=cs3_ins_dc[i]
           
            cbl_cum_daly[i]=cbl_dalys[i]
            cs1_cum_daly[i]=cs1_dalys[i]
            cs2_cum_daly[i]=cs2_dalys[i]
            cs3_cum_daly[i]=cs3_dalys[i]
        else:
            cbl_cum_dc[i]=cbl_cum_dc[i-1]+cbl_ins_dc[i]
            cs1_cum_dc[i]=cs1_cum_dc[i-1]+cs1_ins_dc[i]
            cs2_cum_dc[i]=cs2_cum_dc[i-1]+cs2_ins_dc[i]
            cs3_cum_dc[i]=cs3_cum_dc[i-1]+cs3_ins_dc[i]
            
            cbl_cum_daly[i]=cbl_cum_daly[i-1]+cbl_dalys[i]
            cs1_cum_daly[i]=cs1_cum_daly[i-1]+cs1_dalys[i]
            cs2_cum_daly[i]=cs2_cum_daly[i-1]+cs2_dalys[i]
            cs3_cum_daly[i]=cs3_cum_daly[i-1]+cs3_dalys[i]
        
    cs1_icer=-(cbl_cum_dc-cs1_cum_dc)/(cbl_cum_daly-cs1_cum_daly)
    cs2_icer=-(cbl_cum_dc-cs2_cum_dc)/(cbl_cum_daly-cs2_cum_daly)
    cs3_icer=-(cbl_cum_dc-cs3_cum_dc)/(cbl_cum_daly-cs3_cum_daly)
        
    ## FIX: sanitize inf and nan (happens when DALY diff = 0 in early years)
    cs1_icer=np.nan_to_num(cs1_icer, nan=0.0, posinf=0.0, neginf=0.0)
    cs2_icer=np.nan_to_num(cs2_icer, nan=0.0, posinf=0.0, neginf=0.0)
    cs3_icer=np.nan_to_num(cs3_icer, nan=0.0, posinf=0.0, neginf=0.0)
    
    bl_ins_dc=dmc_psa_bl+imc_psa_bl
    s1_ins_dc=dmc_psa_s1+imc_psa_s1
    s2_ins_dc=dmc_psa_s2+imc_psa_s2
    s3_ins_dc=dmc_psa_s3+imc_psa_s3

    bl_cum_dc=np.zeros(np.shape(bl_ins_dc))
    s1_cum_dc=np.zeros(np.shape(s1_ins_dc))
    s2_cum_dc=np.zeros(np.shape(s2_ins_dc))
    s3_cum_dc=np.zeros(np.shape(s3_ins_dc))
   
    bl_cum_daly=np.zeros(np.shape(bl_dalys))
    s1_cum_daly=np.zeros(np.shape(s1_dalys))
    s2_cum_daly=np.zeros(np.shape(s2_dalys))
    s3_cum_daly=np.zeros(np.shape(s3_dalys))
    
    for i in range(len(bl_ins_dc)):
        if i < 1:
            bl_cum_dc[i,:]=bl_ins_dc[i,:]
            s1_cum_dc[i,:]=s1_ins_dc[i,:]
            s2_cum_dc[i,:]=s2_ins_dc[i,:]
            s3_cum_dc[i,:]=s3_ins_dc[i,:]
           
            bl_cum_daly[i,:]=bl_dalys[i,:]
            s1_cum_daly[i,:]=s1_dalys[i,:]
            s2_cum_daly[i,:]=s2_dalys[i,:]
            s3_cum_daly[i,:]=s3_dalys[i,:]
        else:
            bl_cum_dc[i,:]=bl_cum_dc[i-1,:]+bl_ins_dc[i,:]
            s1_cum_dc[i,:]=s1_cum_dc[i-1,:]+s1_ins_dc[i,:]
            s2_cum_dc[i,:]=s2_cum_dc[i-1,:]+s2_ins_dc[i,:]
            s3_cum_dc[i,:]=s3_cum_dc[i-1,:]+s3_ins_dc[i,:]
            
            bl_cum_daly[i,:]=bl_cum_daly[i-1,:]+bl_dalys[i,:]
            s1_cum_daly[i,:]=s1_cum_daly[i-1,:]+s1_dalys[i,:]
            s2_cum_daly[i,:]=s2_cum_daly[i-1,:]+s2_dalys[i,:]
            s3_cum_daly[i,:]=s3_cum_daly[i-1,:]+s3_dalys[i,:]
            
    s1_icer=-(bl_cum_dc-s1_cum_dc)/(bl_cum_daly-s1_cum_daly)
    s2_icer=-(bl_cum_dc-s2_cum_dc)/(bl_cum_daly-s2_cum_daly)
    s3_icer=-(bl_cum_dc-s3_cum_dc)/(bl_cum_daly-s3_cum_daly)
    
    ## FIX: sanitize inf and nan (happens when DALY diff = 0 in early years)
    s1_icer=np.nan_to_num(s1_icer, nan=0.0, posinf=0.0, neginf=0.0)
    s2_icer=np.nan_to_num(s2_icer, nan=0.0, posinf=0.0, neginf=0.0)
    s3_icer=np.nan_to_num(s3_icer, nan=0.0, posinf=0.0, neginf=0.0)
    
    econ_dict={}
    #Year by year DALYs
    econ_dict["bl_daly"]=bl_dalys
    econ_dict["s1_daly"]=s1_dalys
    econ_dict["s2_daly"]=s2_dalys
    econ_dict["s3_daly"]=s3_dalys
    
    econ_dict["cbl_daly"]=cbl_dalys
    econ_dict["cs1_daly"]=cs1_dalys
    econ_dict["cs2_daly"]=cs2_dalys
    econ_dict["cs3_daly"]=cs3_dalys
    
    #Year by year intervention costs
    econ_dict["bl_intc"]=dmc_psa_bl
    econ_dict["s1_intc"]=dmc_psa_s1
    econ_dict["s2_intc"]=dmc_psa_s2
    econ_dict["s3_intc"]=dmc_psa_s3
    
    econ_dict["cbl_intc"]=dmc_cent_bl
    econ_dict["cs1_intc"]=dmc_cent_s1
    econ_dict["cs2_intc"]=dmc_cent_s2
    econ_dict["cs3_intc"]=dmc_cent_s3
       
    #Year by year medical costs
    econ_dict["bl_medc"]=imc_psa_bl
    econ_dict["s1_medc"]=imc_psa_s1
    econ_dict["s2_medc"]=imc_psa_s2
    econ_dict["s3_medc"]=imc_psa_s3
    
    econ_dict["cbl_medc"]=imc_cent_bl
    econ_dict["cs1_medc"]=imc_cent_s1
    econ_dict["cs2_medc"]=imc_cent_s2
    econ_dict["cs3_medc"]=imc_cent_s3
    
    #Year by year productivity costs
    econ_dict["bl_prod"]=bl_prod
    econ_dict["s1_prod"]=s1_prod
    econ_dict["s2_prod"]=s2_prod
    econ_dict["s3_prod"]=s3_prod
   
    econ_dict["cbl_prod"]=cbl_prod
    econ_dict["cs1_prod"]=cs1_prod
    econ_dict["cs2_prod"]=cs2_prod
    econ_dict["cs3_prod"]=cs3_prod 
   
    #Year by year direct costs (for plot)
    econ_dict["bl_dirc"]=bl_ins_dc
    econ_dict["s1_dirc"]=s1_ins_dc
    econ_dict["s2_dirc"]=s2_ins_dc
    econ_dict["s3_dirc"]=s3_ins_dc
    
    econ_dict["cbl_dirc"]=cbl_ins_dc
    econ_dict["cs1_dirc"]=cs1_ins_dc
    econ_dict["cs2_dirc"]=cs2_ins_dc
    econ_dict["cs3_dirc"]=cs3_ins_dc
    
    #Year by year total costs (just in case)
    econ_dict["bl_tcos"]=bl_ins_tc
    econ_dict["s1_tcos"]=s1_ins_tc
    econ_dict["s2_tcos"]=s2_ins_tc
    econ_dict["s3_tcos"]=s3_ins_tc
    
    econ_dict["cbl_tcos"]=cbl_ins_tc
    econ_dict["cs1_tcos"]=cs1_ins_tc
    econ_dict["cs2_tcos"]=cs2_ins_tc
    econ_dict["cs3_tcos"]=cs3_ins_tc
    
    #Year by year Net Economic Benefit
    econ_dict["s1_neb"]=s1_neb
    econ_dict["s2_neb"]=s2_neb
    econ_dict["s3_neb"]=s3_neb
    
    econ_dict["cs1_neb"]=cs1_neb
    econ_dict["cs2_neb"]=cs2_neb
    econ_dict["cs3_neb"]=cs3_neb
    
    # Year by year ICERs
    econ_dict["s1_icer"]=s1_icer
    econ_dict["s2_icer"]=s2_icer
    econ_dict["s3_icer"]=s3_icer
    
    econ_dict["cs1_icer"]=cs1_icer
    econ_dict["cs2_icer"]=cs2_icer
    econ_dict["cs3_icer"]=cs3_icer
    
    #Cumulative Direct Costs (for global ICERs)
    econ_dict["bl_cumdc"]=bl_cum_dc
    econ_dict["s1_cumdc"]=s1_cum_dc
    econ_dict["s2_cumdc"]=s2_cum_dc
    econ_dict["s3_cumdc"]=s3_cum_dc
    
    econ_dict["cbl_cumdc"]=cbl_cum_dc
    econ_dict["cs1_cumdc"]=cs1_cum_dc
    econ_dict["cs2_cumdc"]=cs2_cum_dc
    econ_dict["cs3_cumdc"]=cs3_cum_dc
    
    
    #Cumulative DALYs (for ICERS)
    econ_dict["bl_cdal"]=bl_cum_daly
    econ_dict["s1_cdal"]=s1_cum_daly
    econ_dict["s2_cdal"]=s2_cum_daly
    econ_dict["s3_cdal"]=s3_cum_daly
    
    econ_dict["cbl_cdal"]=cbl_cum_daly
    econ_dict["cs1_cdal"]=cs1_cum_daly
    econ_dict["cs2_cdal"]=cs2_cum_daly
    econ_dict["cs3_cdal"]=cs3_cum_daly
    
    # FC drug costs (separate output for cost breakdown analysis)
    econ_dict["bl_fcc"]=fc_psa_bl
    econ_dict["s1_fcc"]=fc_psa_s1
    econ_dict["s2_fcc"]=fc_psa_s2
    econ_dict["s3_fcc"]=fc_psa_s3
    econ_dict["cbl_fcc"]=cfc_cost_bl
    econ_dict["cs1_fcc"]=cfc_cost_s1
    econ_dict["cs2_fcc"]=cfc_cost_s2
    econ_dict["cs3_fcc"]=cfc_cost_s3
    
    return econ_dict

#Economic outcomes (central and sampled) for each WHO region, global outcomes calculated post-hoc
afr_econ= econ_analysis (afr_bl_cent, afr_s1_cent, afr_s2_cent, afr_s3_cent,afr_bl_runs, afr_s1_runs, afr_s2_runs, afr_s3_runs, "AFR", wd, "cost and agg data/costs.xlsx", runs, 0.015, 0.015)
amr_econ= econ_analysis (amr_bl_cent, amr_s1_cent, amr_s2_cent, amr_s3_cent, amr_bl_runs, amr_s1_runs, amr_s2_runs, amr_s3_runs,"AMR", wd, "cost and agg data/costs.xlsx", runs, 0.015, 0.015)
emr_econ= econ_analysis (emr_bl_cent, emr_s1_cent, emr_s2_cent, emr_s3_cent, emr_bl_runs, emr_s1_runs, emr_s2_runs, emr_s3_runs,"EMR", wd, "cost and agg data/costs.xlsx", runs, 0.015, 0.015)
eur_econ= econ_analysis (eur_bl_cent, eur_s1_cent, eur_s2_cent, eur_s3_cent,eur_bl_runs, eur_s1_runs, eur_s2_runs, eur_s3_runs, "EUR", wd, "cost and agg data/costs.xlsx", runs, 0.015, 0.015)
sear_econ= econ_analysis (sear_bl_cent, sear_s1_cent, sear_s2_cent, sear_s3_cent, sear_bl_runs, sear_s1_runs, sear_s2_runs, sear_s3_runs,"SEAR", wd, "cost and agg data/costs.xlsx", runs, 0.015, 0.015)
wpr_econ= econ_analysis (wpr_bl_cent, wpr_s1_cent, wpr_s2_cent, wpr_s3_cent, wpr_bl_runs, wpr_s1_runs, wpr_s2_runs, wpr_s3_runs, "WPR", wd, "cost and agg data/costs.xlsx", runs, 0.015, 0.015)

#%% Intervention Coverage and Epidemiological Outcomes Plots
def epi_plots(cbl, cs1, cs2, cs3, bl, s1, s2, s3, wd, calib, reg):
    
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns    
    import numpy as np
    
    plot_bd=pd.read_excel(wd+calib, sheet_name="bd_vax")
    plot_hb3=pd.read_excel(wd+calib, sheet_name="hb3_vax")
    plot_prev=pd.read_excel(wd+calib, sheet_name="pop_prv")
    plot_prev_u5=pd.read_excel(wd+calib, sheet_name="u5_prev")
    plot_death=pd.read_excel(wd+calib, sheet_name="t_deaths")
    plot_hcc=pd.read_excel(wd+calib, sheet_name="hcc_inc")
    
    colours=["#17202A","#66c2a5", "#fc8d62", "#8da0cb"] #black for baseline, then follow qualitative Set2
    
    sns.set_theme(style="whitegrid")
    
    if reg!="global":
    ## Intervention Coverage Plots
        cov_plot=plt.figure(figsize=(15,15))
    #Birth Dose Vaccine Coverage (caveat that coverage of mAVs and HBIG in the period 2022-2050 matches coverage of HepB-BD)
        bd=cov_plot.add_subplot(2,2,1)
        bd.plot(np.arange(1990,2100,1).T, cbl["bd_cov"]*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        #bd.plot(np.arange(1990,2100,1).T, np.percentile(bl["bd_cov"],50, axis=1)*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        bd.fill_between(np.arange(1990,2100,1).T, np.percentile(bl["bd_cov"],2.5, axis=1)*1e2, np.percentile(bl["bd_cov"],97.5, axis=1)*1e2,color=colours[0], alpha=0.2)
        bd.plot(np.arange(1990,2100,1).T, cs1["bd_cov"]*1e2, color=colours[1], alpha=0.8)
        #bd.plot(np.arange(1990,2100,1).T, np.percentile(s1["bd_cov"],50, axis=1)*1e2, color=colours[1], alpha=0.8)
        bd.fill_between(np.arange(1990,2100,1).T, np.percentile(s1["bd_cov"],2.5, axis=1)*1e2, np.percentile(s1["bd_cov"],97.5, axis=1)*1e2,color=colours[1], alpha=0.2)
        bd.plot(np.arange(1990,2100,1).T, cs2["bd_cov"]*1e2, color=colours[2], alpha=0.8)
        #bd.plot(np.arange(1990,2100,1).T, np.percentile(s2["bd_cov"],50, axis=1)*1e2, color=colours[2], alpha=0.8)
        bd.fill_between(np.arange(1990,2100,1).T, np.percentile(s2["bd_cov"],2.5, axis=1)*1e2, np.percentile(s2["bd_cov"],97.5, axis=1)*1e2,color=colours[2], alpha=0.2)
        bd.plot(np.arange(1990,2100,1).T, cs3["bd_cov"]*1e2, color=colours[3], alpha=0.8)
        #bd.plot(np.arange(1990,2100,1).T, np.percentile(s3["bd_cov"],50, axis=1)*1e2, color=colours[3], alpha=0.8)
        bd.fill_between(np.arange(1990,2100,1).T, np.percentile(s3["bd_cov"],2.5, axis=1)*1e2, np.percentile(s3["bd_cov"],97.5, axis=1)*1e2,color=colours[3], alpha=0.2)
        bd.scatter(plot_bd["year"], plot_bd[reg]*1e2)
        bd.set_xlim(2000,2099)
        bd.set_ylim([0,100])
        bd.yaxis.set_major_formatter(mtick.PercentFormatter())
        bd.tick_params(axis="both", which="major", labelsize=14)
        bd.hlines(y=90, xmin=2000, xmax=2099, color="red", alpha=0.6, linestyle=':')
        bd.set_title("Hepatitis B Birth Dose Vaccine", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
        bd.set_ylabel("Coverage", fontsize=16)
    
    #HepB3 coverage
        hb3=cov_plot.add_subplot(2,2,2)
        hb3.plot(np.arange(1990,2100,1).T, cbl["hb3_cov"]*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        hb3.plot(np.arange(1990,2100,1).T, cbl["hb3_cov"]*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        #hb3.plot(np.arange(1990,2100,1).T, np.percentile(bl["hb3_cov"],50, axis=1)*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        hb3.fill_between(np.arange(1990,2100,1).T, np.percentile(bl["hb3_cov"],2.5, axis=1)*1e2, np.percentile(bl["hb3_cov"],97.5, axis=1)*1e2,color=colours[0], alpha=0.2)
        hb3.plot(np.arange(1990,2100,1).T, cs1["hb3_cov"]*1e2, color=colours[1], alpha=0.8)
        #hb3.plot(np.arange(1990,2100,1).T, np.percentile(s1["hb3_cov"],50, axis=1)*1e2, color=colours[1], alpha=0.8)
        hb3.fill_between(np.arange(1990,2100,1).T, np.percentile(s1["hb3_cov"],2.5, axis=1)*1e2, np.percentile(s1["hb3_cov"],97.5, axis=1)*1e2,color=colours[1], alpha=0.2)
        hb3.plot(np.arange(1990,2100,1).T, cs2["hb3_cov"]*1e2, color=colours[2], alpha=0.8)
        #hb3.plot(np.arange(1990,2100,1).T, np.percentile(s2["hb3_cov"],50, axis=1)*1e2, color=colours[2], alpha=0.8)
        hb3.fill_between(np.arange(1990,2100,1).T, np.percentile(s2["hb3_cov"],2.5, axis=1)*1e2, np.percentile(s2["hb3_cov"],97.5, axis=1)*1e2,color=colours[2], alpha=0.2)
        hb3.plot(np.arange(1990,2100,1).T, cs3["hb3_cov"]*1e2, color=colours[3], alpha=0.8)
        #hb3.plot(np.arange(1990,2100,1).T, np.percentile(s3["hb3_cov"],50, axis=1)*1e2, color=colours[3], alpha=0.8)
        hb3.fill_between(np.arange(1990,2100,1).T, np.percentile(s3["hb3_cov"],2.5, axis=1)*1e2, np.percentile(s3["hb3_cov"],97.5, axis=1)*1e2,color=colours[3], alpha=0.2)
        hb3.scatter(plot_bd["year"], plot_hb3[reg]*1e2)
        hb3.set_xlim(2000,2099)
        hb3.set_ylim([0,100])
        hb3.yaxis.set_major_formatter(mtick.PercentFormatter())
        hb3.hlines(y=90, xmin=2000, xmax=2099, color="red", alpha=0.6, linestyle=':')
        hb3.tick_params(axis="both", which="major", labelsize=14)
        hb3.set_title("Infant (three-dose HepB) Vaccine", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
        hb3.set_ylabel("Coverage", fontsize=16)
    
    #Diagnsosis Coverage
        dx=cov_plot.add_subplot(2,2,3)
        dx.plot(np.arange(1990,2100,1).T, cbl["dx_prop"]*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        #dx.plot(np.arange(1990,2100,1).T, np.percentile(bl["dx_prop"],50, axis=1)*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        dx.fill_between(np.arange(1990,2100,1).T, np.percentile(bl["dx_prop"],2.5, axis=1)*1e2, np.percentile(bl["dx_prop"],97.5, axis=1)*1e2,color=colours[0], alpha=0.2)
        dx.plot(np.arange(1990,2100,1).T, cs1["dx_prop"]*1e2, color=colours[1], alpha=0.8)
        #dx.plot(np.arange(1990,2100,1).T, np.percentile(s1["dx_prop"],50, axis=1)*1e2, color=colours[1], alpha=0.8)
        dx.fill_between(np.arange(1990,2100,1).T, np.percentile(s1["dx_prop"],2.5, axis=1)*1e2, np.percentile(s1["dx_prop"],97.5, axis=1)*1e2,color=colours[1], alpha=0.2)
        dx.plot(np.arange(1990,2100,1).T, cs2["dx_prop"]*1e2, color=colours[2], alpha=0.8)
        #dx.plot(np.arange(1990,2100,1).T, np.percentile(s2["dx_prop"],50, axis=1)*1e2, color=colours[2], alpha=0.8)
        dx.fill_between(np.arange(1990,2100,1).T, np.percentile(s2["dx_prop"],2.5, axis=1)*1e2, np.percentile(s2["dx_prop"],97.5, axis=1)*1e2,color=colours[2], alpha=0.2)
        dx.plot(np.arange(1990,2100,1).T, cs3["dx_prop"]*1e2, color=colours[3], alpha=0.8)
        #dx.plot(np.arange(1990,2100,1).T, np.percentile(s3["dx_prop"],50, axis=1)*1e2, color=colours[3], alpha=0.8)
        dx.fill_between(np.arange(1990,2100,1).T, np.percentile(s3["dx_prop"],2.5, axis=1)*1e2, np.percentile(s3["dx_prop"],97.5, axis=1)*1e2,color=colours[3], alpha=0.2)
        dx.hlines(y=90, xmin=2000, xmax=2099, color="red", alpha=0.6, linestyle=':')
        #dx.scatter(2021, plot_care[reg].iloc[3]*1e2, color="blue",marker="X", s=50)
        dx.tick_params(axis="both", which="major", labelsize=14)
        dx.set_xlim([2000,2099])
        dx.set_ylim([0,100])
        dx.yaxis.set_major_formatter(mtick.PercentFormatter())
        dx.set_title("Chronic Hepatitis B (Diagnosed)", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
        dx.set_ylabel("Coverage", fontsize=16)
    
    #Treatment Coverage
        tx=cov_plot.add_subplot(2,2,4)
        tx.plot(np.arange(1990,2100,1).T, cbl["tx_prop"]*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        tx.plot(np.arange(1990,2100,1).T, cs1["tx_prop"]*1e2, color=colours[1], alpha=0.8)
        tx.plot(np.arange(1990,2100,1).T, cs2["tx_prop"]*1e2, color=colours[2], alpha=0.8)
        tx.plot(np.arange(1990,2100,1).T, cs3["tx_prop"]*1e2, color=colours[3], alpha=0.8)
        #tx.plot(np.arange(1990,2100,1).T, np.percentile(bl["tx_prop"],50, axis=1)*1e2, color=colours[0], linestyle="dashed", alpha=0.8)
        #tx.plot(np.arange(1990,2100,1).T, np.percentile(s1["tx_prop"],50, axis=1)*1e2, color=colours[1], alpha=0.8)
        #tx.plot(np.arange(1990,2100,1).T, np.percentile(s2["tx_prop"],50, axis=1)*1e2, color=colours[2], alpha=0.8)
        #tx.plot(np.arange(1990,2100,1).T, np.percentile(s3["tx_prop"],50, axis=1)*1e2, color=colours[3], alpha=0.8)
        tx.fill_between(np.arange(1990,2100,1).T, np.percentile(bl["tx_prop"],2.5, axis=1)*1e2, np.percentile(bl["tx_prop"],97.5, axis=1)*1e2,color=colours[0], alpha=0.2)
        tx.fill_between(np.arange(1990,2100,1).T, np.percentile(s1["tx_prop"],2.5, axis=1)*1e2, np.percentile(s1["tx_prop"],97.5, axis=1)*1e2,color=colours[1], alpha=0.2)
        tx.fill_between(np.arange(1990,2100,1).T, np.percentile(s2["tx_prop"],2.5, axis=1)*1e2, np.percentile(s2["tx_prop"],97.5, axis=1)*1e2,color=colours[2], alpha=0.2)
        tx.fill_between(np.arange(1990,2100,1).T, np.percentile(s3["tx_prop"],2.5, axis=1)*1e2, np.percentile(s3["tx_prop"],97.5, axis=1)*1e2,color=colours[3], alpha=0.2)
        tx.hlines(y=80, xmin=2000, xmax=2099, color="red", alpha=0.6, linestyle=':')
        #tx.scatter(2021, plot_care[reg].iloc[4]*1e2, color="blue", marker="X", s=50) #add marker for calibration tx coverage
        #tx.scatter(2021, np.percentile((bl["tx_prop"][32,:],50)*1e2, color="red", marker="X",s=50)  #add modelled equivalent of treatment(n)/diag(n) in 2021
        tx.set_xlim([2000,2099])
        tx.set_ylim([0,100])
        tx.yaxis.set_major_formatter(mtick.PercentFormatter())
        tx.tick_params(axis="both", which="major", labelsize=14)
        tx.set_title("Chronic Hepatitis B (Eligible on Treatment)", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
        tx.set_ylabel("Coverage", fontsize=16)
        tx.legend(("Status Quo", "FC Slow (15y to 80%)", "FC Mid (10y to 80%)", "FC Fast (5y to 80%)"), fontsize="large", frameon=False, loc="upper left")
    
        #plt.subplots_adjust(top=0.96, bottom=0.08, left=0.055, right=0.959, hspace=0.35, wspace=0.35)
        plt.tight_layout()
        plt.savefig("cov_plots/"+reg+"_coverage plot_PSA.png", dpi=300)
    
    ## Epidemiological/Calibration Plots
    psa_plot=plt.figure(figsize=(15,15))
    
    ## Population Prevalence
    prev=psa_plot.add_subplot(2,2,1)
    prev.plot(np.arange(1990,2100,1).T, cbl["prev"]*1e2,color=colours[0], linestyle="dashed", alpha=0.8)
    #prev.plot(np.arange(1990,2100,1).T, np.percentile(bl["prev"],50, axis=1)*1e2,color=colours[0], linestyle="dashed", alpha=0.8)
    prev.fill_between(np.arange(1990,2100,1).T,np.percentile(bl["prev"],2.5, axis=1)*1e2,np.percentile(bl["prev"],97.5, axis=1)*1e2, color=colours[0], alpha=0.2)
    prev.plot(np.arange(1990,2100,1).T, cs1["prev"]*1e2,color=colours[1], alpha=0.8)
    #prev.plot(np.arange(1990,2100,1).T, np.percentile(s1["prev"],50, axis=1)*1e2,color=colours[1], alpha=0.8)
    prev.fill_between(np.arange(1990,2100,1).T,np.percentile(s1["prev"],2.5, axis=1)*1e2,np.percentile(s1["prev"],97.5, axis=1)*1e2, color=colours[1], alpha=0.2)
    prev.plot(np.arange(1990,2100,1).T, cs2["prev"]*1e2,color=colours[2], alpha=0.8)
    #prev.plot(np.arange(1990,2100,1).T, np.percentile(s2["prev"],50, axis=1)*1e2,color=colours[2], alpha=0.8)
    prev.fill_between(np.arange(1990,2100,1).T,np.percentile(s2["prev"],2.5, axis=1)*1e2,np.percentile(s2["prev"],97.5, axis=1)*1e2, color=colours[2], alpha=0.2)
    prev.plot(np.arange(1990,2100,1).T, cs3["prev"]*1e2,color=colours[3], alpha=0.8)
    #prev.plot(np.arange(1990,2100,1).T, np.percentile(s3["prev"],50, axis=1)*1e2,color=colours[3], alpha=0.8)
    prev.fill_between(np.arange(1990,2100,1).T,np.percentile(s3["prev"],2.5, axis=1)*1e2,np.percentile(s3["prev"],97.5, axis=1)*1e2, color=colours[3], alpha=0.2)
    prev.scatter(plot_prev["year"], plot_prev[reg]*1e2)
    prev.set_ylim([0,(max(np.percentile(bl["prev"],97.5, axis=1)*1e2))+0.5])
    prev.set_xlim(2000,2099)
    prev.yaxis.set_major_formatter(mtick.PercentFormatter())
    prev.tick_params(axis="both", which="major", labelsize=14)
    prev.set_title("Population Prevalence", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    prev.set_ylabel("HBsAg seroprevalence", fontsize=16)
    #plt.grid()
    
    ## Under5y Prevalence
    prev_u5=psa_plot.add_subplot(2,2,2)
    prev_u5.plot(np.arange(1990,2100,1).T, cbl["prev_u5"]*1e2,color=colours[0], linestyle="dashed", alpha=0.8)
    #prev_u5.plot(np.arange(1990,2100,1).T, np.percentile(bl["prev_u5"],50, axis=1)*1e2,color=colours[0], linestyle="dashed", alpha=0.8)
    prev_u5.fill_between(np.arange(1990,2100,1).T,np.percentile(bl["prev_u5"],2.5, axis=1)*1e2,np.percentile(bl["prev_u5"],97.5, axis=1)*1e2, color=colours[0], alpha=0.2)
    prev_u5.plot(np.arange(1990,2100,1).T, cs1["prev_u5"]*1e2,color=colours[1], alpha=0.8)
    #prev_u5.plot(np.arange(1990,2100,1).T, np.percentile(s1["prev_u5"],50, axis=1)*1e2,color=colours[1], alpha=0.8)
    prev_u5.fill_between(np.arange(1990,2100,1).T,np.percentile(s1["prev_u5"],2.5, axis=1)*1e2,np.percentile(s1["prev_u5"],97.5, axis=1)*1e2, color=colours[1], alpha=0.2)
    prev_u5.plot(np.arange(1990,2100,1).T, cs2["prev_u5"]*1e2,color=colours[2], alpha=0.8)
    #prev_u5.plot(np.arange(1990,2100,1).T, np.percentile(s2["prev_u5"],50, axis=1)*1e2,color=colours[2], alpha=0.8)
    prev_u5.fill_between(np.arange(1990,2100,1).T,np.percentile(s2["prev_u5"],2.5, axis=1)*1e2,np.percentile(s2["prev_u5"],97.5, axis=1)*1e2, color=colours[2], alpha=0.2)
    prev_u5.plot(np.arange(1990,2100,1).T, cs3["prev_u5"]*1e2,color=colours[3], alpha=0.8)
    #prev_u5.plot(np.arange(1990,2100,1).T, np.percentile(s3["prev_u5"],50, axis=1)*1e2,color=colours[3], alpha=0.8)
    prev_u5.fill_between(np.arange(1990,2100,1).T,np.percentile(s3["prev_u5"],2.5, axis=1)*1e2,np.percentile(s3["prev_u5"],97.5, axis=1)*1e2, color=colours[3], alpha=0.2)
    prev_u5.scatter(plot_prev_u5["year"], plot_prev_u5[reg]*1e2)
    prev_u5.set_ylim([0,(max(np.percentile(bl["prev_u5"],97.5, axis=1)*1e2))+0.5])
    prev_u5.set_xlim(2000,2099)
    prev_u5.yaxis.set_major_formatter(mtick.PercentFormatter())
    prev_u5.tick_params(axis="both", which="major", labelsize=14)
    prev_u5.set_title("Under 5y Prevalence", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    prev_u5.set_ylabel("HBsAg seroprevalence", fontsize=16)
    #plt.grid()
    
    ## HBV Mortality
    mort=psa_plot.add_subplot(2,2,3)
    mort.plot(np.arange(1990,2100,1).T,cbl["mort"]/1e3,color=colours[0], linestyle="dashed", alpha=0.8)
    #mort.plot(np.arange(1990,2100,1).T, np.percentile(bl["mort"],50, axis=1)/1e3,color=colours[0], linestyle="dashed", alpha=0.8)
    mort.fill_between(np.arange(1990,2100,1).T,np.percentile(bl["mort"],2.5, axis=1)/1e3,np.percentile(bl["mort"],97.5, axis=1)/1e3, color=colours[0], alpha=0.2)
    mort.plot(np.arange(1990,2100,1).T, cs1["mort"]/1e3,color=colours[1], alpha=0.8)
    #mort.plot(np.arange(1990,2100,1).T, np.percentile(s1["mort"],50, axis=1)/1e3,color=colours[1], alpha=0.8)
    mort.fill_between(np.arange(1990,2100,1).T,np.percentile(s1["mort"],2.5, axis=1)/1e3,np.percentile(s1["mort"],97.5, axis=1)/1e3, color=colours[1], alpha=0.2)
    mort.plot(np.arange(1990,2100,1).T, cs2["mort"]/1e3,color=colours[2], alpha=0.8)
    #mort.plot(np.arange(1990,2100,1).T, np.percentile(s2["mort"],50, axis=1)/1e3,color=colours[2], alpha=0.8)
    mort.fill_between(np.arange(1990,2100,1).T,np.percentile(s2["mort"],2.5, axis=1)/1e3,np.percentile(s2["mort"],97.5, axis=1)/1e3, color=colours[2], alpha=0.2)
    mort.plot(np.arange(1990,2100,1).T, cs3["mort"]/1e3,color=colours[3], alpha=0.8)
    #mort.plot(np.arange(1990,2100,1).T, np.percentile(s3["mort"],50, axis=1)/1e3,color=colours[3], alpha=0.8)
    mort.fill_between(np.arange(1990,2100,1).T,np.percentile(s3["mort"],2.5, axis=1)/1e3,np.percentile(s3["mort"],97.5, axis=1)/1e3, color=colours[3], alpha=0.2)
    mort.scatter(plot_death["year"], plot_death[reg]/1e3)
    mort.set_xlim(2000,2099)
    #mort.set_ylim([0, np.percentile(bl["mort"][36:110,:],97.5)+500])
    mort.tick_params(axis="both", which="major", labelsize=14)
    mort.set_title("Hepatitis B Attributable Mortality", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    mort.set_ylabel("Annual Deaths (thousand)", fontsize=16)
    #plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))  
    #plt.grid()
    
    ## HCC incidence 
    hcc=psa_plot.add_subplot(2,2,4)
    hcc.plot(np.arange(1990,2100,1).T, cbl["hcc_inc"]/1e3,color=colours[0], linestyle="dashed", alpha=0.8)
    hcc.plot(np.arange(1990,2100,1).T, cs1["hcc_inc"]/1e3,color=colours[1], alpha=0.8)
    hcc.plot(np.arange(1990,2100,1).T, cs2["hcc_inc"]/1e3,color=colours[2], alpha=0.8)
    hcc.plot(np.arange(1990,2100,1).T, cs3["hcc_inc"]/1e3,color=colours[3], alpha=0.8)
    #hcc.plot(np.arange(1990,2100,1).T, np.percentile(bl["hcc_inc"],50, axis=1)/1e3,color=colours[0], linestyle="dashed", alpha=0.8)
    #hcc.plot(np.arange(1990,2100,1).T, np.percentile(s1["hcc_inc"],50, axis=1)/1e3,color=colours[1], alpha=0.8)
    #hcc.plot(np.arange(1990,2100,1).T, np.percentile(s2["hcc_inc"],50, axis=1)/1e3,color=colours[2], alpha=0.8)
    #hcc.plot(np.arange(1990,2100,1).T, np.percentile(s3["hcc_inc"],50, axis=1)/1e3,color=colours[3], alpha=0.8)
    hcc.fill_between(np.arange(1990,2100,1).T,np.percentile(bl["hcc_inc"],2.5, axis=1)/1e3,np.percentile(bl["hcc_inc"],97.5, axis=1)/1e3, color=colours[0], alpha=0.2)
    hcc.fill_between(np.arange(1990,2100,1).T,np.percentile(s1["hcc_inc"],2.5, axis=1)/1e3,np.percentile(s1["hcc_inc"],97.5, axis=1)/1e3, color=colours[1], alpha=0.2)
    hcc.fill_between(np.arange(1990,2100,1).T,np.percentile(s2["hcc_inc"],2.5, axis=1)/1e3,np.percentile(s2["hcc_inc"],97.5, axis=1)/1e3, color=colours[2], alpha=0.2)
    hcc.fill_between(np.arange(1990,2100,1).T,np.percentile(s3["hcc_inc"],2.5, axis=1)/1e3,np.percentile(s3["hcc_inc"],97.5, axis=1)/1e3, color=colours[3], alpha=0.2)
    hcc.scatter(plot_hcc["year"], plot_hcc[reg]/1e3)
    hcc.set_xlim(2000,2099)
    #hcc.set_ylim([0, np.percentile(bl["hcc_inc"][36:110,:],97.5)+500])
    hcc.tick_params(axis="both", which="major", labelsize=14)
    hcc.set_title("Hepatocellular Carcinoma Incidence", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    hcc.set_ylabel("Annual Cases (thousand)", fontsize=16)
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    hcc.legend(("Status Quo", "FC Slow (15y to 80%)", "FC Mid (10y to 80%)", "FC Fast (5y to 80%)"), fontsize="large", frameon=False) 
    #plt.grid()
    
    #plt.subplots_adjust(top=0.96, bottom=0.04, left=0.065, right=0.98, hspace=0.35, wspace=0.35)
    plt.tight_layout()
    
    plt.savefig("epi_plots/"+reg+"_epi plot PSA.png", dpi=500)
    
    if reg=="global": 
        plt.savefig("epi_plots/"+reg+"_epi plot PSA.pdf", dpi=500)
    
    return print("Figures Generated!")

#Intervention coverage and epidemiology plots for each WHO region, global produced following post-hoc calculations
epi_plots(afr_bl_cent, afr_s1_cent, afr_s2_cent,afr_s3_cent, afr_bl_runs, afr_s1_runs, afr_s2_runs, afr_s3_runs, wd, "cost and agg data/pop_calib.xlsx", "AFR")
epi_plots(amr_bl_cent, amr_s1_cent, amr_s2_cent, amr_s3_cent, amr_bl_runs, amr_s1_runs, amr_s2_runs, amr_s3_runs,wd, "cost and agg data/pop_calib.xlsx", "AMR")
epi_plots(emr_bl_cent, emr_s1_cent, emr_s2_cent, emr_s3_cent, emr_bl_runs, emr_s1_runs, emr_s2_runs, emr_s3_runs,wd, "cost and agg data/pop_calib.xlsx", "EMR")
epi_plots(eur_bl_cent, eur_s1_cent, eur_s2_cent, eur_s3_cent,eur_bl_runs, eur_s1_runs, eur_s2_runs, eur_s3_runs, wd, "cost and agg data/pop_calib.xlsx", "EUR")
epi_plots(sear_bl_cent, sear_s1_cent, sear_s2_cent, sear_s3_cent, sear_bl_runs, sear_s1_runs, sear_s2_runs, sear_s3_runs, wd, "cost and agg data/pop_calib.xlsx", "SEAR")
epi_plots(wpr_bl_cent, wpr_s1_cent, wpr_s2_cent, wpr_s3_cent,wpr_bl_runs, wpr_s1_runs, wpr_s2_runs, wpr_s3_runs, wd, "cost and agg data/pop_calib.xlsx", "WPR")

#%% Plot Economic Outcomes
def econ_plots (econ_dict, reg):
    
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns    
    import numpy as np
    
    ## DIAGNOSTIC: dump ICER / NEB core values to a CSV for debugging
    ## Keep a minimal report so we can see what's actually in the arrays
    try:
        import os
        os.makedirs("econ_plots", exist_ok=True)
        diag = {
            "year": np.arange(1990, 2100, 1),
            "cs1_icer": np.asarray(econ_dict["cs1_icer"]).flatten(),
            "cs2_icer": np.asarray(econ_dict["cs2_icer"]).flatten(),
            "cs3_icer": np.asarray(econ_dict["cs3_icer"]).flatten(),
            "cs1_neb":  np.asarray(econ_dict["cs1_neb"]).flatten(),
            "cs2_neb":  np.asarray(econ_dict["cs2_neb"]).flatten(),
            "cs3_neb":  np.asarray(econ_dict["cs3_neb"]).flatten(),
        }
        # pad/trim to year length
        n = len(diag["year"])
        for k,v in list(diag.items()):
            if len(v) != n:
                diag[k] = v[:n] if len(v) > n else np.concatenate([v, np.zeros(n-len(v))])
        import pandas as _pd
        _pd.DataFrame(diag).to_csv(f"econ_plots/{reg}_icer_neb_debug.csv", index=False)
        # also quick summary to stdout
        print(f"[econ_plots:{reg}] cs1_icer range @ 2040-2099: "
              f"[{np.nanmin(diag['cs1_icer'][50:110]):.3e}, {np.nanmax(diag['cs1_icer'][50:110]):.3e}] "
              f"non-zero count = {np.sum(diag['cs1_icer'][50:110]!=0)}")
        print(f"[econ_plots:{reg}] cs1_neb range @ 2026-2099: "
              f"[{np.nanmin(diag['cs1_neb'][36:110]):.3e}, {np.nanmax(diag['cs1_neb'][36:110]):.3e}] "
              f"non-zero count = {np.sum(diag['cs1_neb'][36:110]!=0)}")
    except Exception as _e:
        print(f"[econ_plots:{reg}] diagnostic dump failed: {_e}")
    
    colours=["#17202A","#66c2a5", "#fc8d62", "#8da0cb"] #black for baseline, then follow qualitative Set2
    
    sns.set_theme(style="whitegrid")
    
    econ_plot=plt.figure(figsize=(15,15))
    ## DALYs
    daly=econ_plot.add_subplot(2,2,1)
    daly.plot(np.arange(1990,2100,1).T, econ_dict["cbl_daly"]/1e6,color=colours[0], linestyle="dashed", alpha=0.8)
    daly.plot(np.arange(1990,2100,1).T, econ_dict["cs1_daly"]/1e6,color=colours[1], alpha=0.8)
    daly.plot(np.arange(1990,2100,1).T, econ_dict["cs2_daly"]/1e6,color=colours[2], alpha=0.8)
    daly.plot(np.arange(1990,2100,1).T, econ_dict["cs3_daly"]/1e6,color=colours[3], alpha=0.8)
    # daly.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["bl_daly"],50, axis=1)/1e6,color=colours[0], linestyle="dashed", alpha=0.8)
    # daly.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["s1_daly"],50, axis=1)/1e6,color=colours[1], alpha=0.8)
    # daly.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["s2_daly"],50, axis=1)/1e6,color=colours[2], alpha=0.8)
    # daly.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["s3_daly"],50, axis=1)/1e6,color=colours[3], alpha=0.8)
    daly.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["bl_daly"],2.5, axis=1)/1e6,np.percentile(econ_dict["bl_daly"],97.5, axis=1)/1e6, color=colours[0], alpha=0.2)
    daly.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["s1_daly"],2.5, axis=1)/1e6,np.percentile(econ_dict["s1_daly"],97.5, axis=1)/1e6, color=colours[1], alpha=0.2)
    daly.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["s2_daly"],2.5, axis=1)/1e6,np.percentile(econ_dict["s2_daly"],97.5, axis=1)/1e6, color=colours[2], alpha=0.2)
    daly.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["s3_daly"],2.5, axis=1)/1e6,np.percentile(econ_dict["s3_daly"],97.5, axis=1)/1e6, color=colours[3], alpha=0.2)
    daly.set_xlim(2026,2099)
    daly.set_title("Disability Adjusted Life Years", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    daly.set_ylabel("DALYs (million)", fontsize=16)
    daly.tick_params(axis="both", which="major", labelsize=14)
    daly.legend(("Status Quo", "FC Slow (15y to 80%)", "FC Mid (10y to 80%)", "FC Fast (5y to 80%)"), fontsize="large", frameon=False) 

    
    ## Direct Costs
    dc=econ_plot.add_subplot(2,2,2)
    dc.plot(np.arange(1990,2100,1).T, econ_dict["cbl_dirc"]/1e9,color=colours[0], linestyle="dashed", alpha=0.8)
    #dc.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["bl_dirc"],50, axis=1)/1e9,color=colours[0], linestyle="dashed", alpha=0.8)
    dc.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["bl_dirc"],2.5, axis=1)/1e9,np.percentile(econ_dict["bl_dirc"],97.5, axis=1)/1e9, color=colours[0], alpha=0.2)
    dc.plot(np.arange(1990,2100,1).T, econ_dict["cs1_dirc"]/1e9,color=colours[1], alpha=0.8)
    #dc.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["s1_dirc"],50, axis=1)/1e9,color=colours[1], alpha=0.8)
    dc.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["s1_dirc"],2.5, axis=1)/1e9,np.percentile(econ_dict["s1_dirc"],97.5, axis=1)/1e9, color=colours[1], alpha=0.2)
    dc.plot(np.arange(1990,2100,1).T, econ_dict["cs2_dirc"]/1e9,color=colours[2], alpha=0.8)
    #dc.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["s2_dirc"],50, axis=1)/1e9,color=colours[2], alpha=0.8)
    dc.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["s2_dirc"],2.5, axis=1)/1e9,np.percentile(econ_dict["s2_dirc"],97.5, axis=1)/1e9, color=colours[2], alpha=0.2)
    dc.plot(np.arange(1990,2100,1).T, econ_dict["cs3_dirc"]/1e9,color=colours[3], alpha=0.8)
    #dc.plot(np.arange(1990,2100,1).T, np.percentile(econ_dict["s3_dirc"],50, axis=1)/1e9,color=colours[3], alpha=0.8)
    dc.fill_between(np.arange(1990,2100,1).T,np.percentile(econ_dict["s3_dirc"],2.5, axis=1)/1e9,np.percentile(econ_dict["s3_dirc"],97.5, axis=1)/1e9, color=colours[3], alpha=0.2)
    dc.set_xlim(2026,2099)
    dc.tick_params(axis="both", which="major", labelsize=14)
    dc.set_title("Healthcare Costs", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    dc.set_ylabel("US$ (billion)", fontsize=16)
    
    
    ## ICERs
    ## FIX: extend x-range to 2030-2099 and auto-clip y-axis (early years have huge ICERs from tiny DALY diffs)
    icer=econ_plot.add_subplot(2,2,3)
    icer_years = np.arange(2030, 2100, 1)  # skip 2025-2029 where DALY diffs ~0 -> ICER blows up
    icer_idx_start, icer_idx_end = 40, 110  # year index: 2030 = 40, 2099 = 109 (inclusive end 110)
    
    # Sanitize before plotting: ensure no inf/nan contaminate the visualisation
    def _clean(arr):
        a = np.asarray(arr, dtype=float).flatten()
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    cs1_icer_clean = _clean(econ_dict["cs1_icer"])
    cs2_icer_clean = _clean(econ_dict["cs2_icer"])
    cs3_icer_clean = _clean(econ_dict["cs3_icer"])
    s1_icer_clean = np.nan_to_num(np.asarray(econ_dict["s1_icer"]), nan=0.0, posinf=0.0, neginf=0.0)
    s2_icer_clean = np.nan_to_num(np.asarray(econ_dict["s2_icer"]), nan=0.0, posinf=0.0, neginf=0.0)
    s3_icer_clean = np.nan_to_num(np.asarray(econ_dict["s3_icer"]), nan=0.0, posinf=0.0, neginf=0.0)
    
    icer.plot(icer_years, cs1_icer_clean[icer_idx_start:icer_idx_end]/1e3,color=colours[1], alpha=0.8, label="FC Slow")
    icer.fill_between(icer_years,np.percentile(s1_icer_clean[icer_idx_start:icer_idx_end,:],2.5, axis=1)/1e3,np.percentile(s1_icer_clean[icer_idx_start:icer_idx_end,:],97.5, axis=1)/1e3, color=colours[1], alpha=0.2)
    icer.plot(icer_years, cs2_icer_clean[icer_idx_start:icer_idx_end]/1e3,color=colours[2], alpha=0.8, label="FC Mid")
    icer.fill_between(icer_years,np.percentile(s2_icer_clean[icer_idx_start:icer_idx_end,:],2.5, axis=1)/1e3,np.percentile(s2_icer_clean[icer_idx_start:icer_idx_end,:],97.5, axis=1)/1e3, color=colours[2], alpha=0.2)
    icer.plot(icer_years, cs3_icer_clean[icer_idx_start:icer_idx_end]/1e3,color=colours[3], alpha=0.8, label="FC Fast")
    icer.fill_between(icer_years,np.percentile(s3_icer_clean[icer_idx_start:icer_idx_end,:],2.5, axis=1)/1e3,np.percentile(s3_icer_clean[icer_idx_start:icer_idx_end,:],97.5, axis=1)/1e3, color=colours[3], alpha=0.2)
    icer.set_xlim(2030,2099)
    
    # Robust auto y-range: use cleaned central values from 2040 onwards
    icer_stack = np.concatenate([
        cs1_icer_clean[50:110]/1e3,
        cs2_icer_clean[50:110]/1e3,
        cs3_icer_clean[50:110]/1e3,
    ])
    icer_stack = icer_stack[np.isfinite(icer_stack)]
    if len(icer_stack) > 0:
        lo = float(np.min(icer_stack))
        hi = float(np.max(icer_stack))
        # Ensure a non-degenerate range even if all values are close to 0
        if hi - lo < 1e-6:
            # Data is essentially constant - set a default informative range around it
            mid = (hi + lo) / 2
            icer.set_ylim(mid - 1.0, max(mid + 1.0, 100.0))  # show at least ±1 or up to 100k
        else:
            span = hi - lo
            pad = max(1.0, 0.1*span)
            icer.set_ylim(lo - pad, hi + pad)
    
    icer.tick_params(axis="both", which="major", labelsize=14)
    icer.set_title("ICER", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    icer.set_ylabel("US$ (thousand) per DALY averted", fontsize=16)
    
    ## Net Economic Benefit
    ## FIX: add zero reference line and use data-driven y-range
    neb=econ_plot.add_subplot(2,2,4)
    cs1_neb_clean = _clean(econ_dict["cs1_neb"])
    cs2_neb_clean = _clean(econ_dict["cs2_neb"])
    cs3_neb_clean = _clean(econ_dict["cs3_neb"])
    s1_neb_clean = np.nan_to_num(np.asarray(econ_dict["s1_neb"]), nan=0.0, posinf=0.0, neginf=0.0)
    s2_neb_clean = np.nan_to_num(np.asarray(econ_dict["s2_neb"]), nan=0.0, posinf=0.0, neginf=0.0)
    s3_neb_clean = np.nan_to_num(np.asarray(econ_dict["s3_neb"]), nan=0.0, posinf=0.0, neginf=0.0)
    
    neb.plot(np.arange(2026,2100,1).T, cs1_neb_clean[36:110]/1e9,color=colours[1], alpha=0.8, label="FC Slow")
    neb.fill_between(np.arange(2026,2100,1).T,np.percentile(s1_neb_clean[36:110,:],2.5, axis=1)/1e9,np.percentile(s1_neb_clean[36:110,:],97.5, axis=1)/1e9, color=colours[1], alpha=0.2)
    neb.plot(np.arange(2026,2100,1).T,cs2_neb_clean[36:110]/1e9,color=colours[2], alpha=0.8, label="FC Mid")
    neb.fill_between(np.arange(2026,2100,1).T,np.percentile(s2_neb_clean[36:110,:],2.5, axis=1)/1e9,np.percentile(s2_neb_clean[36:110,:],97.5, axis=1)/1e9, color=colours[2], alpha=0.2)
    neb.plot(np.arange(2026,2100,1).T, cs3_neb_clean[36:110]/1e9,color=colours[3], alpha=0.8, label="FC Fast")
    neb.fill_between(np.arange(2026,2100,1).T,np.percentile(s3_neb_clean[36:110,:],2.5, axis=1)/1e9,np.percentile(s3_neb_clean[36:110,:],97.5, axis=1)/1e9, color=colours[3], alpha=0.2)
    neb.axhline(y=0, color="grey", linestyle=":", linewidth=1, alpha=0.7)  # zero reference
    neb.set_xlim(2026,2099)
    
    # Robust auto y-range
    neb_stack = np.concatenate([
        cs1_neb_clean[36:110]/1e9,
        cs2_neb_clean[36:110]/1e9,
        cs3_neb_clean[36:110]/1e9,
    ])
    neb_stack = neb_stack[np.isfinite(neb_stack)]
    if len(neb_stack) > 0:
        lo = float(np.min(neb_stack))
        hi = float(np.max(neb_stack))
        if hi - lo < 1e-6:
            mid = (hi + lo) / 2
            neb.set_ylim(mid - 1.0, max(mid + 1.0, 10.0))
        else:
            span = hi - lo
            pad = max(1.0, 0.15*span)
            neb.set_ylim(lo - pad, hi + pad)
    
    neb.tick_params(axis="both", which="major", labelsize=14)
    neb.set_title("Net Economic Benefit", fontdict={"fontsize":20, "fontweight":"roman", "family": _FONT_FAMILY})
    neb.set_ylabel("US$ (billion)", fontsize=16)
    
    #plt.subplots_adjust(top=0.96, bottom=0.04, left=0.065, right=0.98, hspace=0.35, wspace=0.35)
    plt.tight_layout()
    
    if reg=="global":
        plt.savefig("econ_plots/"+reg+"_econ plot PSA.pdf", dpi=500)
    
    plt.savefig("econ_plots/"+reg+"_econ plot PSA.png", dpi=300)
    
        
    
    return print("Figure Generated!")

#Economic outcome plots for each WHO region, global produced following post-hoc calculations
econ_plots (afr_econ, "AFR")
econ_plots (amr_econ, "AMR")
econ_plots (emr_econ, "EMR")
econ_plots (eur_econ, "EUR")
econ_plots (sear_econ, "SEAR")
econ_plots (wpr_econ, "WPR")

#%% Post-hoc global outcome calculations and plots
#Generate weights for prevalence aggregation (central) 
afr_bl_cent["prev_w"]=afr_bl_cent["prev"]*afr_bl_cent["pop"]
afr_s1_cent["prev_w"]=afr_s1_cent["prev"]*afr_s1_cent["pop"]
afr_s2_cent["prev_w"]=afr_s2_cent["prev"]*afr_s2_cent["pop"]
afr_s3_cent["prev_w"]=afr_s3_cent["prev"]*afr_s3_cent["pop"]

afr_bl_cent["prev_u5_w"]=afr_bl_cent["prev_u5"]*afr_bl_cent["pop_u5"]
afr_s1_cent["prev_u5_w"]=afr_s1_cent["prev_u5"]*afr_s1_cent["pop_u5"]
afr_s2_cent["prev_u5_w"]=afr_s2_cent["prev_u5"]*afr_s2_cent["pop_u5"]
afr_s3_cent["prev_u5_w"]=afr_s3_cent["prev_u5"]*afr_s3_cent["pop_u5"]

amr_bl_cent["prev_w"]=amr_bl_cent["prev"]*amr_bl_cent["pop"]
amr_s1_cent["prev_w"]=amr_s1_cent["prev"]*amr_s1_cent["pop"]
amr_s2_cent["prev_w"]=amr_s2_cent["prev"]*amr_s2_cent["pop"]
amr_s3_cent["prev_w"]=amr_s3_cent["prev"]*amr_s3_cent["pop"]

amr_bl_cent["prev_u5_w"]=amr_bl_cent["prev_u5"]*amr_bl_cent["pop_u5"]
amr_s1_cent["prev_u5_w"]=amr_s1_cent["prev_u5"]*amr_s1_cent["pop_u5"]
amr_s2_cent["prev_u5_w"]=amr_s2_cent["prev_u5"]*amr_s2_cent["pop_u5"]
amr_s3_cent["prev_u5_w"]=amr_s3_cent["prev_u5"]*amr_s3_cent["pop_u5"]

emr_bl_cent["prev_w"]=emr_bl_cent["prev"]*emr_bl_cent["pop"]
emr_s1_cent["prev_w"]=emr_s1_cent["prev"]*emr_s1_cent["pop"]
emr_s2_cent["prev_w"]=emr_s2_cent["prev"]*emr_s2_cent["pop"]
emr_s3_cent["prev_w"]=emr_s3_cent["prev"]*emr_s3_cent["pop"]

emr_bl_cent["prev_u5_w"]=emr_bl_cent["prev_u5"]*emr_bl_cent["pop_u5"]
emr_s1_cent["prev_u5_w"]=emr_s1_cent["prev_u5"]*emr_s1_cent["pop_u5"]
emr_s2_cent["prev_u5_w"]=emr_s2_cent["prev_u5"]*emr_s2_cent["pop_u5"]
emr_s3_cent["prev_u5_w"]=emr_s3_cent["prev_u5"]*emr_s3_cent["pop_u5"]

eur_bl_cent["prev_w"]=eur_bl_cent["prev"]*eur_bl_cent["pop"]
eur_s1_cent["prev_w"]=eur_s1_cent["prev"]*eur_s1_cent["pop"]
eur_s2_cent["prev_w"]=eur_s2_cent["prev"]*eur_s2_cent["pop"]
eur_s3_cent["prev_w"]=eur_s3_cent["prev"]*eur_s3_cent["pop"]

eur_bl_cent["prev_u5_w"]=eur_bl_cent["prev_u5"]*eur_bl_cent["pop_u5"]
eur_s1_cent["prev_u5_w"]=eur_s1_cent["prev_u5"]*eur_s1_cent["pop_u5"]
eur_s2_cent["prev_u5_w"]=eur_s2_cent["prev_u5"]*eur_s2_cent["pop_u5"]
eur_s3_cent["prev_u5_w"]=eur_s3_cent["prev_u5"]*eur_s3_cent["pop_u5"]

sear_bl_cent["prev_w"]=sear_bl_cent["prev"]*sear_bl_cent["pop"]
sear_s1_cent["prev_w"]=sear_s1_cent["prev"]*sear_s1_cent["pop"]
sear_s2_cent["prev_w"]=sear_s2_cent["prev"]*sear_s2_cent["pop"]
sear_s3_cent["prev_w"]=sear_s3_cent["prev"]*sear_s3_cent["pop"]

sear_bl_cent["prev_u5_w"]=sear_bl_cent["prev_u5"]*sear_bl_cent["pop_u5"]
sear_s1_cent["prev_u5_w"]=sear_s1_cent["prev_u5"]*sear_s1_cent["pop_u5"]
sear_s2_cent["prev_u5_w"]=sear_s2_cent["prev_u5"]*sear_s2_cent["pop_u5"]
sear_s3_cent["prev_u5_w"]=sear_s3_cent["prev_u5"]*sear_s3_cent["pop_u5"]

wpr_bl_cent["prev_w"]=wpr_bl_cent["prev"]*wpr_bl_cent["pop"]
wpr_s1_cent["prev_w"]=wpr_s1_cent["prev"]*wpr_s1_cent["pop"]
wpr_s2_cent["prev_w"]=wpr_s2_cent["prev"]*wpr_s2_cent["pop"]
wpr_s3_cent["prev_w"]=wpr_s3_cent["prev"]*wpr_s3_cent["pop"]

wpr_bl_cent["prev_u5_w"]=wpr_bl_cent["prev_u5"]*wpr_bl_cent["pop_u5"]
wpr_s1_cent["prev_u5_w"]=wpr_s1_cent["prev_u5"]*wpr_s1_cent["pop_u5"]
wpr_s2_cent["prev_u5_w"]=wpr_s2_cent["prev_u5"]*wpr_s2_cent["pop_u5"]
wpr_s3_cent["prev_u5_w"]=wpr_s3_cent["prev_u5"]*wpr_s3_cent["pop_u5"]


#Generate weights for prevalence aggregation (sampled) 
afr_bl_runs["prev_w"]=afr_bl_runs["prev"]*afr_bl_runs["pop"]
afr_s1_runs["prev_w"]=afr_s1_runs["prev"]*afr_s1_runs["pop"]
afr_s2_runs["prev_w"]=afr_s2_runs["prev"]*afr_s2_runs["pop"]
afr_s3_runs["prev_w"]=afr_s3_runs["prev"]*afr_s3_runs["pop"]

afr_bl_runs["prev_u5_w"]=afr_bl_runs["prev_u5"]*afr_bl_runs["pop_u5"]
afr_s1_runs["prev_u5_w"]=afr_s1_runs["prev_u5"]*afr_s1_runs["pop_u5"]
afr_s2_runs["prev_u5_w"]=afr_s2_runs["prev_u5"]*afr_s2_runs["pop_u5"]
afr_s3_runs["prev_u5_w"]=afr_s3_runs["prev_u5"]*afr_s3_runs["pop_u5"]

amr_bl_runs["prev_w"]=amr_bl_runs["prev"]*amr_bl_runs["pop"]
amr_s1_runs["prev_w"]=amr_s1_runs["prev"]*amr_s1_runs["pop"]
amr_s2_runs["prev_w"]=amr_s2_runs["prev"]*amr_s2_runs["pop"]
amr_s3_runs["prev_w"]=amr_s3_runs["prev"]*amr_s3_runs["pop"]

amr_bl_runs["prev_u5_w"]=amr_bl_runs["prev_u5"]*amr_bl_runs["pop_u5"]
amr_s1_runs["prev_u5_w"]=amr_s1_runs["prev_u5"]*amr_s1_runs["pop_u5"]
amr_s2_runs["prev_u5_w"]=amr_s2_runs["prev_u5"]*amr_s2_runs["pop_u5"]
amr_s3_runs["prev_u5_w"]=amr_s3_runs["prev_u5"]*amr_s3_runs["pop_u5"]

emr_bl_runs["prev_w"]=emr_bl_runs["prev"]*emr_bl_runs["pop"]
emr_s1_runs["prev_w"]=emr_s1_runs["prev"]*emr_s1_runs["pop"]
emr_s2_runs["prev_w"]=emr_s2_runs["prev"]*emr_s2_runs["pop"]
emr_s3_runs["prev_w"]=emr_s3_runs["prev"]*emr_s3_runs["pop"]

emr_bl_runs["prev_u5_w"]=emr_bl_runs["prev_u5"]*emr_bl_runs["pop_u5"]
emr_s1_runs["prev_u5_w"]=emr_s1_runs["prev_u5"]*emr_s1_runs["pop_u5"]
emr_s2_runs["prev_u5_w"]=emr_s2_runs["prev_u5"]*emr_s2_runs["pop_u5"]
emr_s3_runs["prev_u5_w"]=emr_s3_runs["prev_u5"]*emr_s3_runs["pop_u5"]

eur_bl_runs["prev_w"]=eur_bl_runs["prev"]*eur_bl_runs["pop"]
eur_s1_runs["prev_w"]=eur_s1_runs["prev"]*eur_s1_runs["pop"]
eur_s2_runs["prev_w"]=eur_s2_runs["prev"]*eur_s2_runs["pop"]
eur_s3_runs["prev_w"]=eur_s3_runs["prev"]*eur_s3_runs["pop"]

eur_bl_runs["prev_u5_w"]=eur_bl_runs["prev_u5"]*eur_bl_runs["pop_u5"]
eur_s1_runs["prev_u5_w"]=eur_s1_runs["prev_u5"]*eur_s1_runs["pop_u5"]
eur_s2_runs["prev_u5_w"]=eur_s2_runs["prev_u5"]*eur_s2_runs["pop_u5"]
eur_s3_runs["prev_u5_w"]=eur_s3_runs["prev_u5"]*eur_s3_runs["pop_u5"]

sear_bl_runs["prev_w"]=sear_bl_runs["prev"]*sear_bl_runs["pop"]
sear_s1_runs["prev_w"]=sear_s1_runs["prev"]*sear_s1_runs["pop"]
sear_s2_runs["prev_w"]=sear_s2_runs["prev"]*sear_s2_runs["pop"]
sear_s3_runs["prev_w"]=sear_s3_runs["prev"]*sear_s3_runs["pop"]

sear_bl_runs["prev_u5_w"]=sear_bl_runs["prev_u5"]*sear_bl_runs["pop_u5"]
sear_s1_runs["prev_u5_w"]=sear_s1_runs["prev_u5"]*sear_s1_runs["pop_u5"]
sear_s2_runs["prev_u5_w"]=sear_s2_runs["prev_u5"]*sear_s2_runs["pop_u5"]
sear_s3_runs["prev_u5_w"]=sear_s3_runs["prev_u5"]*sear_s3_runs["pop_u5"]

wpr_bl_runs["prev_w"]=wpr_bl_runs["prev"]*wpr_bl_runs["pop"]
wpr_s1_runs["prev_w"]=wpr_s1_runs["prev"]*wpr_s1_runs["pop"]
wpr_s2_runs["prev_w"]=wpr_s2_runs["prev"]*wpr_s2_runs["pop"]
wpr_s3_runs["prev_w"]=wpr_s3_runs["prev"]*wpr_s3_runs["pop"]

wpr_bl_runs["prev_u5_w"]=wpr_bl_runs["prev_u5"]*wpr_bl_runs["pop_u5"]
wpr_s1_runs["prev_u5_w"]=wpr_s1_runs["prev_u5"]*wpr_s1_runs["pop_u5"]
wpr_s2_runs["prev_u5_w"]=wpr_s2_runs["prev_u5"]*wpr_s2_runs["pop_u5"]
wpr_s3_runs["prev_u5_w"]=wpr_s3_runs["prev_u5"]*wpr_s3_runs["pop_u5"]

## Generate Required Data for Global Epi Outcomes (central)
global_bl_cent={}
global_bl_cent["prev"]=(afr_bl_cent["prev_w"]+amr_bl_cent["prev_w"]+emr_bl_cent["prev_w"]+eur_bl_cent["prev_w"]+sear_bl_cent["prev_w"]+wpr_bl_cent["prev_w"])/(afr_bl_cent["pop"]+amr_bl_cent["pop"]+emr_bl_cent["pop"]+eur_bl_cent["pop"]+sear_bl_cent["pop"]+wpr_bl_cent["pop"])
global_bl_cent["prev_u5"]=(afr_bl_cent["prev_u5_w"]+amr_bl_cent["prev_u5_w"]+emr_bl_cent["prev_u5_w"]+eur_bl_cent["prev_u5_w"]+sear_bl_cent["prev_u5_w"]+wpr_bl_cent["prev_u5_w"])/(afr_bl_cent["pop_u5"]+amr_bl_cent["pop_u5"]+emr_bl_cent["pop_u5"]+eur_bl_cent["pop_u5"]+sear_bl_cent["pop_u5"]+wpr_bl_cent["pop_u5"])
global_bl_cent["hcc_inc"]=afr_bl_cent["hcc_inc"]+amr_bl_cent["hcc_inc"]+emr_bl_cent["hcc_inc"]+eur_bl_cent["hcc_inc"]+sear_bl_cent["hcc_inc"]+wpr_bl_cent["hcc_inc"]
global_bl_cent["mort"]=afr_bl_cent["mort"]+amr_bl_cent["mort"]+emr_bl_cent["mort"]+eur_bl_cent["mort"]+sear_bl_cent["mort"]+wpr_bl_cent["mort"]
global_bl_cent["chb_inc"]=afr_bl_cent["chb_inc"]+amr_bl_cent["chb_inc"]+emr_bl_cent["chb_inc"]+eur_bl_cent["chb_inc"]+sear_bl_cent["chb_inc"]+wpr_bl_cent["chb_inc"]
total_pop = afr_bl_cent["pop"]+amr_bl_cent["pop"]+emr_bl_cent["pop"]+eur_bl_cent["pop"]+sear_bl_cent["pop"]+wpr_bl_cent["pop"]
global_bl_cent["bd_cov"] = (afr_bl_cent["bd_cov"]*afr_bl_cent["pop"]+amr_bl_cent["bd_cov"]*amr_bl_cent["pop"]+emr_bl_cent["bd_cov"]*emr_bl_cent["pop"]+eur_bl_cent["bd_cov"]*eur_bl_cent["pop"]+sear_bl_cent["bd_cov"]*sear_bl_cent["pop"]+wpr_bl_cent["bd_cov"]*wpr_bl_cent["pop"])/total_pop
global_bl_cent["hb3_cov"] = (afr_bl_cent["hb3_cov"]*afr_bl_cent["pop"]+amr_bl_cent["hb3_cov"]*amr_bl_cent["pop"]+emr_bl_cent["hb3_cov"]*emr_bl_cent["pop"]+eur_bl_cent["hb3_cov"]*eur_bl_cent["pop"]+sear_bl_cent["hb3_cov"]*sear_bl_cent["pop"]+wpr_bl_cent["hb3_cov"]*wpr_bl_cent["pop"])/total_pop
global_bl_cent["dx_prop"] = (afr_bl_cent["dx_prop"]*afr_bl_cent["pop"]+amr_bl_cent["dx_prop"]*amr_bl_cent["pop"]+emr_bl_cent["dx_prop"]*emr_bl_cent["pop"]+eur_bl_cent["dx_prop"]*eur_bl_cent["pop"]+sear_bl_cent["dx_prop"]*sear_bl_cent["pop"]+wpr_bl_cent["dx_prop"]*wpr_bl_cent["pop"])/total_pop
global_bl_cent["tx_prop"] = (afr_bl_cent["tx_prop"]*afr_bl_cent["pop"]+amr_bl_cent["tx_prop"]*amr_bl_cent["pop"]+emr_bl_cent["tx_prop"]*emr_bl_cent["pop"]+eur_bl_cent["tx_prop"]*eur_bl_cent["pop"]+sear_bl_cent["tx_prop"]*sear_bl_cent["pop"]+wpr_bl_cent["tx_prop"]*wpr_bl_cent["pop"])/total_pop

global_s1_cent={}
global_s1_cent["prev"]=(afr_s1_cent["prev_w"]+amr_s1_cent["prev_w"]+emr_s1_cent["prev_w"]+eur_s1_cent["prev_w"]+sear_s1_cent["prev_w"]+wpr_s1_cent["prev_w"])/(afr_s1_cent["pop"]+amr_s1_cent["pop"]+emr_s1_cent["pop"]+eur_s1_cent["pop"]+sear_s1_cent["pop"]+wpr_s1_cent["pop"])
global_s1_cent["prev_u5"]=(afr_s1_cent["prev_u5_w"]+amr_s1_cent["prev_u5_w"]+emr_s1_cent["prev_u5_w"]+eur_s1_cent["prev_u5_w"]+sear_s1_cent["prev_u5_w"]+wpr_s1_cent["prev_u5_w"])/(afr_s1_cent["pop_u5"]+amr_s1_cent["pop_u5"]+emr_s1_cent["pop_u5"]+eur_s1_cent["pop_u5"]+sear_s1_cent["pop_u5"]+wpr_s1_cent["pop_u5"])
global_s1_cent["hcc_inc"]=afr_s1_cent["hcc_inc"]+amr_s1_cent["hcc_inc"]+emr_s1_cent["hcc_inc"]+eur_s1_cent["hcc_inc"]+sear_s1_cent["hcc_inc"]+wpr_s1_cent["hcc_inc"]
global_s1_cent["mort"]=afr_s1_cent["mort"]+amr_s1_cent["mort"]+emr_s1_cent["mort"]+eur_s1_cent["mort"]+sear_s1_cent["mort"]+wpr_s1_cent["mort"]
global_s1_cent["chb_inc"]=afr_s1_cent["chb_inc"]+amr_s1_cent["chb_inc"]+emr_s1_cent["chb_inc"]+eur_s1_cent["chb_inc"]+sear_s1_cent["chb_inc"]+wpr_s1_cent["chb_inc"]
total_pop_s1 = afr_s1_cent["pop"]+amr_s1_cent["pop"]+emr_s1_cent["pop"]+eur_s1_cent["pop"]+sear_s1_cent["pop"]+wpr_s1_cent["pop"]
global_s1_cent["bd_cov"] = (afr_s1_cent["bd_cov"]*afr_s1_cent["pop"]+amr_s1_cent["bd_cov"]*amr_s1_cent["pop"]+emr_s1_cent["bd_cov"]*emr_s1_cent["pop"]+eur_s1_cent["bd_cov"]*eur_s1_cent["pop"]+sear_s1_cent["bd_cov"]*sear_s1_cent["pop"]+wpr_s1_cent["bd_cov"]*wpr_s1_cent["pop"])/total_pop_s1
global_s1_cent["hb3_cov"] = (afr_s1_cent["hb3_cov"]*afr_s1_cent["pop"]+amr_s1_cent["hb3_cov"]*amr_s1_cent["pop"]+emr_s1_cent["hb3_cov"]*emr_s1_cent["pop"]+eur_s1_cent["hb3_cov"]*eur_s1_cent["pop"]+sear_s1_cent["hb3_cov"]*sear_s1_cent["pop"]+wpr_s1_cent["hb3_cov"]*wpr_s1_cent["pop"])/total_pop_s1
global_s1_cent["dx_prop"] = (afr_s1_cent["dx_prop"]*afr_s1_cent["pop"]+amr_s1_cent["dx_prop"]*amr_s1_cent["pop"]+emr_s1_cent["dx_prop"]*emr_s1_cent["pop"]+eur_s1_cent["dx_prop"]*eur_s1_cent["pop"]+sear_s1_cent["dx_prop"]*sear_s1_cent["pop"]+wpr_s1_cent["dx_prop"]*wpr_s1_cent["pop"])/total_pop_s1
global_s1_cent["tx_prop"] = (afr_s1_cent["tx_prop"]*afr_s1_cent["pop"]+amr_s1_cent["tx_prop"]*amr_s1_cent["pop"]+emr_s1_cent["tx_prop"]*emr_s1_cent["pop"]+eur_s1_cent["tx_prop"]*eur_s1_cent["pop"]+sear_s1_cent["tx_prop"]*sear_s1_cent["pop"]+wpr_s1_cent["tx_prop"]*wpr_s1_cent["pop"])/total_pop_s1

global_s2_cent={}
global_s2_cent["prev"]=(afr_s2_cent["prev_w"]+amr_s2_cent["prev_w"]+emr_s2_cent["prev_w"]+eur_s2_cent["prev_w"]+sear_s2_cent["prev_w"]+wpr_s2_cent["prev_w"])/(afr_s2_cent["pop"]+amr_s2_cent["pop"]+emr_s2_cent["pop"]+eur_s2_cent["pop"]+sear_s2_cent["pop"]+wpr_s2_cent["pop"])
global_s2_cent["prev_u5"]=(afr_s2_cent["prev_u5_w"]+amr_s2_cent["prev_u5_w"]+emr_s2_cent["prev_u5_w"]+eur_s2_cent["prev_u5_w"]+sear_s2_cent["prev_u5_w"]+wpr_s2_cent["prev_u5_w"])/(afr_s2_cent["pop_u5"]+amr_s2_cent["pop_u5"]+emr_s2_cent["pop_u5"]+eur_s2_cent["pop_u5"]+sear_s2_cent["pop_u5"]+wpr_s2_cent["pop_u5"])
global_s2_cent["hcc_inc"]=afr_s2_cent["hcc_inc"]+amr_s2_cent["hcc_inc"]+emr_s2_cent["hcc_inc"]+eur_s2_cent["hcc_inc"]+sear_s2_cent["hcc_inc"]+wpr_s2_cent["hcc_inc"]
global_s2_cent["mort"]=afr_s2_cent["mort"]+amr_s2_cent["mort"]+emr_s2_cent["mort"]+eur_s2_cent["mort"]+sear_s2_cent["mort"]+wpr_s2_cent["mort"]
global_s2_cent["chb_inc"]=afr_s2_cent["chb_inc"]+amr_s2_cent["chb_inc"]+emr_s2_cent["chb_inc"]+eur_s2_cent["chb_inc"]+sear_s2_cent["chb_inc"]+wpr_s2_cent["chb_inc"]
total_pop_s2 = afr_s2_cent["pop"]+amr_s2_cent["pop"]+emr_s2_cent["pop"]+eur_s2_cent["pop"]+sear_s2_cent["pop"]+wpr_s2_cent["pop"]
global_s2_cent["bd_cov"] = (afr_s2_cent["bd_cov"]*afr_s2_cent["pop"]+amr_s2_cent["bd_cov"]*amr_s2_cent["pop"]+emr_s2_cent["bd_cov"]*emr_s2_cent["pop"]+eur_s2_cent["bd_cov"]*eur_s2_cent["pop"]+sear_s2_cent["bd_cov"]*sear_s2_cent["pop"]+wpr_s2_cent["bd_cov"]*wpr_s2_cent["pop"])/total_pop_s2
global_s2_cent["hb3_cov"] = (afr_s2_cent["hb3_cov"]*afr_s2_cent["pop"]+amr_s2_cent["hb3_cov"]*amr_s2_cent["pop"]+emr_s2_cent["hb3_cov"]*emr_s2_cent["pop"]+eur_s2_cent["hb3_cov"]*eur_s2_cent["pop"]+sear_s2_cent["hb3_cov"]*sear_s2_cent["pop"]+wpr_s2_cent["hb3_cov"]*wpr_s2_cent["pop"])/total_pop_s2
global_s2_cent["dx_prop"] = (afr_s2_cent["dx_prop"]*afr_s2_cent["pop"]+amr_s2_cent["dx_prop"]*amr_s2_cent["pop"]+emr_s2_cent["dx_prop"]*emr_s2_cent["pop"]+eur_s2_cent["dx_prop"]*eur_s2_cent["pop"]+sear_s2_cent["dx_prop"]*sear_s2_cent["pop"]+wpr_s2_cent["dx_prop"]*wpr_s2_cent["pop"])/total_pop_s2
global_s2_cent["tx_prop"] = (afr_s2_cent["tx_prop"]*afr_s2_cent["pop"]+amr_s2_cent["tx_prop"]*amr_s2_cent["pop"]+emr_s2_cent["tx_prop"]*emr_s2_cent["pop"]+eur_s2_cent["tx_prop"]*eur_s2_cent["pop"]+sear_s2_cent["tx_prop"]*sear_s2_cent["pop"]+wpr_s2_cent["tx_prop"]*wpr_s2_cent["pop"])/total_pop_s2

global_s3_cent={}
global_s3_cent["prev"]=(afr_s3_cent["prev_w"]+amr_s3_cent["prev_w"]+emr_s3_cent["prev_w"]+eur_s3_cent["prev_w"]+sear_s3_cent["prev_w"]+wpr_s3_cent["prev_w"])/(afr_s3_cent["pop"]+amr_s3_cent["pop"]+emr_s3_cent["pop"]+eur_s3_cent["pop"]+sear_s3_cent["pop"]+wpr_s3_cent["pop"])
global_s3_cent["prev_u5"]=(afr_s3_cent["prev_u5_w"]+amr_s3_cent["prev_u5_w"]+emr_s3_cent["prev_u5_w"]+eur_s3_cent["prev_u5_w"]+sear_s3_cent["prev_u5_w"]+wpr_s3_cent["prev_u5_w"])/(afr_s3_cent["pop_u5"]+amr_s3_cent["pop_u5"]+emr_s3_cent["pop_u5"]+eur_s3_cent["pop_u5"]+sear_s3_cent["pop_u5"]+wpr_s3_cent["pop_u5"])
global_s3_cent["hcc_inc"]=afr_s3_cent["hcc_inc"]+amr_s3_cent["hcc_inc"]+emr_s3_cent["hcc_inc"]+eur_s3_cent["hcc_inc"]+sear_s3_cent["hcc_inc"]+wpr_s3_cent["hcc_inc"]
global_s3_cent["mort"]=afr_s3_cent["mort"]+amr_s3_cent["mort"]+emr_s3_cent["mort"]+eur_s3_cent["mort"]+sear_s3_cent["mort"]+wpr_s3_cent["mort"]
global_s3_cent["chb_inc"]=afr_s3_cent["chb_inc"]+amr_s3_cent["chb_inc"]+emr_s3_cent["chb_inc"]+eur_s3_cent["chb_inc"]+sear_s3_cent["chb_inc"]+wpr_s3_cent["chb_inc"]
total_pop_s3 = afr_s3_cent["pop"]+amr_s3_cent["pop"]+emr_s3_cent["pop"]+eur_s3_cent["pop"]+sear_s3_cent["pop"]+wpr_s3_cent["pop"]
global_s3_cent["bd_cov"] = (afr_s3_cent["bd_cov"]*afr_s3_cent["pop"]+amr_s3_cent["bd_cov"]*amr_s3_cent["pop"]+emr_s3_cent["bd_cov"]*emr_s3_cent["pop"]+eur_s3_cent["bd_cov"]*eur_s3_cent["pop"]+sear_s3_cent["bd_cov"]*sear_s3_cent["pop"]+wpr_s3_cent["bd_cov"]*wpr_s3_cent["pop"])/total_pop_s3
global_s3_cent["hb3_cov"] = (afr_s3_cent["hb3_cov"]*afr_s3_cent["pop"]+amr_s3_cent["hb3_cov"]*amr_s3_cent["pop"]+emr_s3_cent["hb3_cov"]*emr_s3_cent["pop"]+eur_s3_cent["hb3_cov"]*eur_s3_cent["pop"]+sear_s3_cent["hb3_cov"]*sear_s3_cent["pop"]+wpr_s3_cent["hb3_cov"]*wpr_s3_cent["pop"])/total_pop_s3
global_s3_cent["dx_prop"] = (afr_s3_cent["dx_prop"]*afr_s3_cent["pop"]+amr_s3_cent["dx_prop"]*amr_s3_cent["pop"]+emr_s3_cent["dx_prop"]*emr_s3_cent["pop"]+eur_s3_cent["dx_prop"]*eur_s3_cent["pop"]+sear_s3_cent["dx_prop"]*sear_s3_cent["pop"]+wpr_s3_cent["dx_prop"]*wpr_s3_cent["pop"])/total_pop_s3
global_s3_cent["tx_prop"] = (afr_s3_cent["tx_prop"]*afr_s3_cent["pop"]+amr_s3_cent["tx_prop"]*amr_s3_cent["pop"]+emr_s3_cent["tx_prop"]*emr_s3_cent["pop"]+eur_s3_cent["tx_prop"]*eur_s3_cent["pop"]+sear_s3_cent["tx_prop"]*sear_s3_cent["pop"]+wpr_s3_cent["tx_prop"]*wpr_s3_cent["pop"])/total_pop_s3

## Generate Required Data for Global Epi Outcomes (sampled)
global_bl_runs={}
global_bl_runs["prev"]=(afr_bl_runs["prev_w"]+amr_bl_runs["prev_w"]+emr_bl_runs["prev_w"]+eur_bl_runs["prev_w"]+sear_bl_runs["prev_w"]+wpr_bl_runs["prev_w"])/(afr_bl_runs["pop"]+amr_bl_runs["pop"]+emr_bl_runs["pop"]+eur_bl_runs["pop"]+sear_bl_runs["pop"]+wpr_bl_runs["pop"])
global_bl_runs["prev_u5"]=(afr_bl_runs["prev_u5_w"]+amr_bl_runs["prev_u5_w"]+emr_bl_runs["prev_u5_w"]+eur_bl_runs["prev_u5_w"]+sear_bl_runs["prev_u5_w"]+wpr_bl_runs["prev_u5_w"])/(afr_bl_runs["pop_u5"]+amr_bl_runs["pop_u5"]+emr_bl_runs["pop_u5"]+eur_bl_runs["pop_u5"]+sear_bl_runs["pop_u5"]+wpr_bl_runs["pop_u5"])
global_bl_runs["hcc_inc"]=afr_bl_runs["hcc_inc"]+amr_bl_runs["hcc_inc"]+emr_bl_runs["hcc_inc"]+eur_bl_runs["hcc_inc"]+sear_bl_runs["hcc_inc"]+wpr_bl_runs["hcc_inc"]
global_bl_runs["mort"]=afr_bl_runs["mort"]+amr_bl_runs["mort"]+emr_bl_runs["mort"]+eur_bl_runs["mort"]+sear_bl_runs["mort"]+wpr_bl_runs["mort"]
global_bl_runs["chb_inc"]=afr_bl_runs["chb_inc"]+amr_bl_runs["chb_inc"]+emr_bl_runs["chb_inc"]+eur_bl_runs["chb_inc"]+sear_bl_runs["chb_inc"]+wpr_bl_runs["chb_inc"]
# Coverage metrics: weighted average across regions (sampled arrays: shape = time x runs)
total_pop_r = afr_bl_runs["pop"]+amr_bl_runs["pop"]+emr_bl_runs["pop"]+eur_bl_runs["pop"]+sear_bl_runs["pop"]+wpr_bl_runs["pop"]
global_bl_runs["bd_cov"] = (afr_bl_runs["bd_cov"]*afr_bl_runs["pop"]+amr_bl_runs["bd_cov"]*amr_bl_runs["pop"]+emr_bl_runs["bd_cov"]*emr_bl_runs["pop"]+eur_bl_runs["bd_cov"]*eur_bl_runs["pop"]+sear_bl_runs["bd_cov"]*sear_bl_runs["pop"]+wpr_bl_runs["bd_cov"]*wpr_bl_runs["pop"])/total_pop_r
global_bl_runs["hb3_cov"] = (afr_bl_runs["hb3_cov"]*afr_bl_runs["pop"]+amr_bl_runs["hb3_cov"]*amr_bl_runs["pop"]+emr_bl_runs["hb3_cov"]*emr_bl_runs["pop"]+eur_bl_runs["hb3_cov"]*eur_bl_runs["pop"]+sear_bl_runs["hb3_cov"]*sear_bl_runs["pop"]+wpr_bl_runs["hb3_cov"]*wpr_bl_runs["pop"])/total_pop_r
global_bl_runs["dx_prop"] = (afr_bl_runs["dx_prop"]*afr_bl_runs["pop"]+amr_bl_runs["dx_prop"]*amr_bl_runs["pop"]+emr_bl_runs["dx_prop"]*emr_bl_runs["pop"]+eur_bl_runs["dx_prop"]*eur_bl_runs["pop"]+sear_bl_runs["dx_prop"]*sear_bl_runs["pop"]+wpr_bl_runs["dx_prop"]*wpr_bl_runs["pop"])/total_pop_r
global_bl_runs["tx_prop"] = (afr_bl_runs["tx_prop"]*afr_bl_runs["pop"]+amr_bl_runs["tx_prop"]*amr_bl_runs["pop"]+emr_bl_runs["tx_prop"]*emr_bl_runs["pop"]+eur_bl_runs["tx_prop"]*eur_bl_runs["pop"]+sear_bl_runs["tx_prop"]*sear_bl_runs["pop"]+wpr_bl_runs["tx_prop"]*wpr_bl_runs["pop"])/total_pop_r

global_s1_runs={}
global_s1_runs["prev"]=(afr_s1_runs["prev_w"]+amr_s1_runs["prev_w"]+emr_s1_runs["prev_w"]+eur_s1_runs["prev_w"]+sear_s1_runs["prev_w"]+wpr_s1_runs["prev_w"])/(afr_s1_runs["pop"]+amr_s1_runs["pop"]+emr_s1_runs["pop"]+eur_s1_runs["pop"]+sear_s1_runs["pop"]+wpr_s1_runs["pop"])
global_s1_runs["prev_u5"]=(afr_s1_runs["prev_u5_w"]+amr_s1_runs["prev_u5_w"]+emr_s1_runs["prev_u5_w"]+eur_s1_runs["prev_u5_w"]+sear_s1_runs["prev_u5_w"]+wpr_s1_runs["prev_u5_w"])/(afr_s1_runs["pop_u5"]+amr_s1_runs["pop_u5"]+emr_s1_runs["pop_u5"]+eur_s1_runs["pop_u5"]+sear_s1_runs["pop_u5"]+wpr_s1_runs["pop_u5"])
global_s1_runs["hcc_inc"]=afr_s1_runs["hcc_inc"]+amr_s1_runs["hcc_inc"]+emr_s1_runs["hcc_inc"]+eur_s1_runs["hcc_inc"]+sear_s1_runs["hcc_inc"]+wpr_s1_runs["hcc_inc"]
global_s1_runs["mort"]=afr_s1_runs["mort"]+amr_s1_runs["mort"]+emr_s1_runs["mort"]+eur_s1_runs["mort"]+sear_s1_runs["mort"]+wpr_s1_runs["mort"]
global_s1_runs["chb_inc"]=afr_s1_runs["chb_inc"]+amr_s1_runs["chb_inc"]+emr_s1_runs["chb_inc"]+eur_s1_runs["chb_inc"]+sear_s1_runs["chb_inc"]+wpr_s1_runs["chb_inc"]
total_pop_s1r = afr_s1_runs["pop"]+amr_s1_runs["pop"]+emr_s1_runs["pop"]+eur_s1_runs["pop"]+sear_s1_runs["pop"]+wpr_s1_runs["pop"]
global_s1_runs["bd_cov"] = (afr_s1_runs["bd_cov"]*afr_s1_runs["pop"]+amr_s1_runs["bd_cov"]*amr_s1_runs["pop"]+emr_s1_runs["bd_cov"]*emr_s1_runs["pop"]+eur_s1_runs["bd_cov"]*eur_s1_runs["pop"]+sear_s1_runs["bd_cov"]*sear_s1_runs["pop"]+wpr_s1_runs["bd_cov"]*wpr_s1_runs["pop"])/total_pop_s1r
global_s1_runs["hb3_cov"] = (afr_s1_runs["hb3_cov"]*afr_s1_runs["pop"]+amr_s1_runs["hb3_cov"]*amr_s1_runs["pop"]+emr_s1_runs["hb3_cov"]*emr_s1_runs["pop"]+eur_s1_runs["hb3_cov"]*eur_s1_runs["pop"]+sear_s1_runs["hb3_cov"]*sear_s1_runs["pop"]+wpr_s1_runs["hb3_cov"]*wpr_s1_runs["pop"])/total_pop_s1r
global_s1_runs["dx_prop"] = (afr_s1_runs["dx_prop"]*afr_s1_runs["pop"]+amr_s1_runs["dx_prop"]*amr_s1_runs["pop"]+emr_s1_runs["dx_prop"]*emr_s1_runs["pop"]+eur_s1_runs["dx_prop"]*eur_s1_runs["pop"]+sear_s1_runs["dx_prop"]*sear_s1_runs["pop"]+wpr_s1_runs["dx_prop"]*wpr_s1_runs["pop"])/total_pop_s1r
global_s1_runs["tx_prop"] = (afr_s1_runs["tx_prop"]*afr_s1_runs["pop"]+amr_s1_runs["tx_prop"]*amr_s1_runs["pop"]+emr_s1_runs["tx_prop"]*emr_s1_runs["pop"]+eur_s1_runs["tx_prop"]*eur_s1_runs["pop"]+sear_s1_runs["tx_prop"]*sear_s1_runs["pop"]+wpr_s1_runs["tx_prop"]*wpr_s1_runs["pop"])/total_pop_s1r

global_s2_runs={}
global_s2_runs["prev"]=(afr_s2_runs["prev_w"]+amr_s2_runs["prev_w"]+emr_s2_runs["prev_w"]+eur_s2_runs["prev_w"]+sear_s2_runs["prev_w"]+wpr_s2_runs["prev_w"])/(afr_s2_runs["pop"]+amr_s2_runs["pop"]+emr_s2_runs["pop"]+eur_s2_runs["pop"]+sear_s2_runs["pop"]+wpr_s2_runs["pop"])
global_s2_runs["prev_u5"]=(afr_s2_runs["prev_u5_w"]+amr_s2_runs["prev_u5_w"]+emr_s2_runs["prev_u5_w"]+eur_s2_runs["prev_u5_w"]+sear_s2_runs["prev_u5_w"]+wpr_s2_runs["prev_u5_w"])/(afr_s2_runs["pop_u5"]+amr_s2_runs["pop_u5"]+emr_s2_runs["pop_u5"]+eur_s2_runs["pop_u5"]+sear_s2_runs["pop_u5"]+wpr_s2_runs["pop_u5"])
global_s2_runs["hcc_inc"]=afr_s2_runs["hcc_inc"]+amr_s2_runs["hcc_inc"]+emr_s2_runs["hcc_inc"]+eur_s2_runs["hcc_inc"]+sear_s2_runs["hcc_inc"]+wpr_s2_runs["hcc_inc"]
global_s2_runs["mort"]=afr_s2_runs["mort"]+amr_s2_runs["mort"]+emr_s2_runs["mort"]+eur_s2_runs["mort"]+sear_s2_runs["mort"]+wpr_s2_runs["mort"]
global_s2_runs["chb_inc"]=afr_s2_runs["chb_inc"]+amr_s2_runs["chb_inc"]+emr_s2_runs["chb_inc"]+eur_s2_runs["chb_inc"]+sear_s2_runs["chb_inc"]+wpr_s2_runs["chb_inc"]
total_pop_s2r = afr_s2_runs["pop"]+amr_s2_runs["pop"]+emr_s2_runs["pop"]+eur_s2_runs["pop"]+sear_s2_runs["pop"]+wpr_s2_runs["pop"]
global_s2_runs["bd_cov"] = (afr_s2_runs["bd_cov"]*afr_s2_runs["pop"]+amr_s2_runs["bd_cov"]*amr_s2_runs["pop"]+emr_s2_runs["bd_cov"]*emr_s2_runs["pop"]+eur_s2_runs["bd_cov"]*eur_s2_runs["pop"]+sear_s2_runs["bd_cov"]*sear_s2_runs["pop"]+wpr_s2_runs["bd_cov"]*wpr_s2_runs["pop"])/total_pop_s2r
global_s2_runs["hb3_cov"] = (afr_s2_runs["hb3_cov"]*afr_s2_runs["pop"]+amr_s2_runs["hb3_cov"]*amr_s2_runs["pop"]+emr_s2_runs["hb3_cov"]*emr_s2_runs["pop"]+eur_s2_runs["hb3_cov"]*eur_s2_runs["pop"]+sear_s2_runs["hb3_cov"]*sear_s2_runs["pop"]+wpr_s2_runs["hb3_cov"]*wpr_s2_runs["pop"])/total_pop_s2r
global_s2_runs["dx_prop"] = (afr_s2_runs["dx_prop"]*afr_s2_runs["pop"]+amr_s2_runs["dx_prop"]*amr_s2_runs["pop"]+emr_s2_runs["dx_prop"]*emr_s2_runs["pop"]+eur_s2_runs["dx_prop"]*eur_s2_runs["pop"]+sear_s2_runs["dx_prop"]*sear_s2_runs["pop"]+wpr_s2_runs["dx_prop"]*wpr_s2_runs["pop"])/total_pop_s2r
global_s2_runs["tx_prop"] = (afr_s2_runs["tx_prop"]*afr_s2_runs["pop"]+amr_s2_runs["tx_prop"]*amr_s2_runs["pop"]+emr_s2_runs["tx_prop"]*emr_s2_runs["pop"]+eur_s2_runs["tx_prop"]*eur_s2_runs["pop"]+sear_s2_runs["tx_prop"]*sear_s2_runs["pop"]+wpr_s2_runs["tx_prop"]*wpr_s2_runs["pop"])/total_pop_s2r

global_s3_runs={}
global_s3_runs["prev"]=(afr_s3_runs["prev_w"]+amr_s3_runs["prev_w"]+emr_s3_runs["prev_w"]+eur_s3_runs["prev_w"]+sear_s3_runs["prev_w"]+wpr_s3_runs["prev_w"])/(afr_s3_runs["pop"]+amr_s3_runs["pop"]+emr_s3_runs["pop"]+eur_s3_runs["pop"]+sear_s3_runs["pop"]+wpr_s3_runs["pop"])
global_s3_runs["prev_u5"]=(afr_s3_runs["prev_u5_w"]+amr_s3_runs["prev_u5_w"]+emr_s3_runs["prev_u5_w"]+eur_s3_runs["prev_u5_w"]+sear_s3_runs["prev_u5_w"]+wpr_s3_runs["prev_u5_w"])/(afr_s3_runs["pop_u5"]+amr_s3_runs["pop_u5"]+emr_s3_runs["pop_u5"]+eur_s3_runs["pop_u5"]+sear_s3_runs["pop_u5"]+wpr_s3_runs["pop_u5"])
global_s3_runs["hcc_inc"]=afr_s3_runs["hcc_inc"]+amr_s3_runs["hcc_inc"]+emr_s3_runs["hcc_inc"]+eur_s3_runs["hcc_inc"]+sear_s3_runs["hcc_inc"]+wpr_s3_runs["hcc_inc"]
global_s3_runs["mort"]=afr_s3_runs["mort"]+amr_s3_runs["mort"]+emr_s3_runs["mort"]+eur_s3_runs["mort"]+sear_s3_runs["mort"]+wpr_s3_runs["mort"]
global_s3_runs["chb_inc"]=afr_s3_runs["chb_inc"]+amr_s3_runs["chb_inc"]+emr_s3_runs["chb_inc"]+eur_s3_runs["chb_inc"]+sear_s3_runs["chb_inc"]+wpr_s3_runs["chb_inc"]
total_pop_s3r = afr_s3_runs["pop"]+amr_s3_runs["pop"]+emr_s3_runs["pop"]+eur_s3_runs["pop"]+sear_s3_runs["pop"]+wpr_s3_runs["pop"]
global_s3_runs["bd_cov"] = (afr_s3_runs["bd_cov"]*afr_s3_runs["pop"]+amr_s3_runs["bd_cov"]*amr_s3_runs["pop"]+emr_s3_runs["bd_cov"]*emr_s3_runs["pop"]+eur_s3_runs["bd_cov"]*eur_s3_runs["pop"]+sear_s3_runs["bd_cov"]*sear_s3_runs["pop"]+wpr_s3_runs["bd_cov"]*wpr_s3_runs["pop"])/total_pop_s3r
global_s3_runs["hb3_cov"] = (afr_s3_runs["hb3_cov"]*afr_s3_runs["pop"]+amr_s3_runs["hb3_cov"]*amr_s3_runs["pop"]+emr_s3_runs["hb3_cov"]*emr_s3_runs["pop"]+eur_s3_runs["hb3_cov"]*eur_s3_runs["pop"]+sear_s3_runs["hb3_cov"]*sear_s3_runs["pop"]+wpr_s3_runs["hb3_cov"]*wpr_s3_runs["pop"])/total_pop_s3r
global_s3_runs["dx_prop"] = (afr_s3_runs["dx_prop"]*afr_s3_runs["pop"]+amr_s3_runs["dx_prop"]*amr_s3_runs["pop"]+emr_s3_runs["dx_prop"]*emr_s3_runs["pop"]+eur_s3_runs["dx_prop"]*eur_s3_runs["pop"]+sear_s3_runs["dx_prop"]*sear_s3_runs["pop"]+wpr_s3_runs["dx_prop"]*wpr_s3_runs["pop"])/total_pop_s3r
global_s3_runs["tx_prop"] = (afr_s3_runs["tx_prop"]*afr_s3_runs["pop"]+amr_s3_runs["tx_prop"]*amr_s3_runs["pop"]+emr_s3_runs["tx_prop"]*emr_s3_runs["pop"]+eur_s3_runs["tx_prop"]*eur_s3_runs["pop"]+sear_s3_runs["tx_prop"]*sear_s3_runs["pop"]+wpr_s3_runs["tx_prop"]*wpr_s3_runs["pop"])/total_pop_s3r


epi_plots(global_bl_cent, global_s1_cent, global_s2_cent, global_s3_cent,global_bl_runs, global_s1_runs, global_s2_runs, global_s3_runs, wd, "cost and agg data/pop_calib.xlsx", "global")

## Generate Required Data for Global Economic Outcomes
global_econ={}

## Add Intervention Costs
global_econ["cbl_intc"]=afr_econ["cbl_intc"]+amr_econ["cbl_intc"]+emr_econ["cbl_intc"]+eur_econ["cbl_intc"]+sear_econ["cbl_intc"]+wpr_econ["cbl_intc"]
global_econ["cs1_intc"]=afr_econ["cs1_intc"]+amr_econ["cs1_intc"]+emr_econ["cs1_intc"]+eur_econ["cs1_intc"]+sear_econ["cs1_intc"]+wpr_econ["cs1_intc"]
global_econ["cs2_intc"]=afr_econ["cs2_intc"]+amr_econ["cs2_intc"]+emr_econ["cs2_intc"]+eur_econ["cs2_intc"]+sear_econ["cs2_intc"]+wpr_econ["cs2_intc"]
global_econ["cs3_intc"]=afr_econ["cs3_intc"]+amr_econ["cs3_intc"]+emr_econ["cs3_intc"]+eur_econ["cs3_intc"]+sear_econ["cs3_intc"]+wpr_econ["cs3_intc"]

global_econ["bl_intc"]=afr_econ["bl_intc"]+amr_econ["bl_intc"]+emr_econ["bl_intc"]+eur_econ["bl_intc"]+sear_econ["bl_intc"]+wpr_econ["bl_intc"]
global_econ["s1_intc"]=afr_econ["s1_intc"]+amr_econ["s1_intc"]+emr_econ["s1_intc"]+eur_econ["s1_intc"]+sear_econ["s1_intc"]+wpr_econ["s1_intc"]
global_econ["s2_intc"]=afr_econ["s2_intc"]+amr_econ["s2_intc"]+emr_econ["s2_intc"]+eur_econ["s2_intc"]+sear_econ["s2_intc"]+wpr_econ["s2_intc"]
global_econ["s3_intc"]=afr_econ["s3_intc"]+amr_econ["s3_intc"]+emr_econ["s3_intc"]+eur_econ["s3_intc"]+sear_econ["s3_intc"]+wpr_econ["s3_intc"]
## Add medical costs
global_econ["cbl_medc"]=afr_econ["cbl_medc"]+amr_econ["cbl_medc"]+emr_econ["cbl_medc"]+eur_econ["cbl_medc"]+sear_econ["cbl_medc"]+wpr_econ["cbl_medc"]
global_econ["cs1_medc"]=afr_econ["cs1_medc"]+amr_econ["cs1_medc"]+emr_econ["cs1_medc"]+eur_econ["cs1_medc"]+sear_econ["cs1_medc"]+wpr_econ["cs1_medc"]
global_econ["cs2_medc"]=afr_econ["cs2_medc"]+amr_econ["cs2_medc"]+emr_econ["cs2_medc"]+eur_econ["cs2_medc"]+sear_econ["cs2_medc"]+wpr_econ["cs2_medc"]
global_econ["cs3_medc"]=afr_econ["cs3_medc"]+amr_econ["cs3_medc"]+emr_econ["cs3_medc"]+eur_econ["cs3_medc"]+sear_econ["cs3_medc"]+wpr_econ["cs3_medc"]

global_econ["bl_medc"]=afr_econ["bl_medc"]+amr_econ["bl_medc"]+emr_econ["bl_medc"]+eur_econ["bl_medc"]+sear_econ["bl_medc"]+wpr_econ["bl_medc"]
global_econ["s1_medc"]=afr_econ["s1_medc"]+amr_econ["s1_medc"]+emr_econ["s1_medc"]+eur_econ["s1_medc"]+sear_econ["s1_medc"]+wpr_econ["s1_medc"]
global_econ["s2_medc"]=afr_econ["s2_medc"]+amr_econ["s2_medc"]+emr_econ["s2_medc"]+eur_econ["s2_medc"]+sear_econ["s2_medc"]+wpr_econ["s2_medc"]
global_econ["s3_medc"]=afr_econ["s3_medc"]+amr_econ["s3_medc"]+emr_econ["s3_medc"]+eur_econ["s3_medc"]+sear_econ["s3_medc"]+wpr_econ["s3_medc"]
## Add productivity Loss
global_econ["cbl_prod"]=afr_econ["cbl_prod"]+amr_econ["cbl_prod"]+emr_econ["cbl_prod"]+eur_econ["cbl_prod"]+sear_econ["cbl_prod"]+wpr_econ["cbl_prod"]
global_econ["cs1_prod"]=afr_econ["cs1_prod"]+amr_econ["cs1_prod"]+emr_econ["cs1_prod"]+eur_econ["cs1_prod"]+sear_econ["cs1_prod"]+wpr_econ["cs1_prod"]
global_econ["cs2_prod"]=afr_econ["cs2_prod"]+amr_econ["cs2_prod"]+emr_econ["cs2_prod"]+eur_econ["cs2_prod"]+sear_econ["cs2_prod"]+wpr_econ["cs2_prod"]
global_econ["cs3_prod"]=afr_econ["cs3_prod"]+amr_econ["cs3_prod"]+emr_econ["cs3_prod"]+eur_econ["cs3_prod"]+sear_econ["cs3_prod"]+wpr_econ["cs3_prod"]

global_econ["bl_prod"]=afr_econ["bl_prod"]+amr_econ["bl_prod"]+emr_econ["bl_prod"]+eur_econ["bl_prod"]+sear_econ["bl_prod"]+wpr_econ["bl_prod"]
global_econ["s1_prod"]=afr_econ["s1_prod"]+amr_econ["s1_prod"]+emr_econ["s1_prod"]+eur_econ["s1_prod"]+sear_econ["s1_prod"]+wpr_econ["s1_prod"]
global_econ["s2_prod"]=afr_econ["s2_prod"]+amr_econ["s2_prod"]+emr_econ["s2_prod"]+eur_econ["s2_prod"]+sear_econ["s2_prod"]+wpr_econ["s2_prod"]
global_econ["s3_prod"]=afr_econ["s3_prod"]+amr_econ["s3_prod"]+emr_econ["s3_prod"]+eur_econ["s3_prod"]+sear_econ["s3_prod"]+wpr_econ["s3_prod"]

global_econ["cbl_daly"]=afr_econ["cbl_daly"]+amr_econ["cbl_daly"]+emr_econ["cbl_daly"]+eur_econ["cbl_daly"]+sear_econ["cbl_daly"]+wpr_econ["cbl_daly"]
global_econ["cs1_daly"]=afr_econ["cs1_daly"]+amr_econ["cs1_daly"]+emr_econ["cs1_daly"]+eur_econ["cs1_daly"]+sear_econ["cs1_daly"]+wpr_econ["cs1_daly"]
global_econ["cs2_daly"]=afr_econ["cs2_daly"]+amr_econ["cs2_daly"]+emr_econ["cs2_daly"]+eur_econ["cs2_daly"]+sear_econ["cs2_daly"]+wpr_econ["cs2_daly"]
global_econ["cs3_daly"]=afr_econ["cs3_daly"]+amr_econ["cs3_daly"]+emr_econ["cs3_daly"]+eur_econ["cs3_daly"]+sear_econ["cs3_daly"]+wpr_econ["cs3_daly"]

global_econ["bl_daly"]=afr_econ["bl_daly"]+amr_econ["bl_daly"]+emr_econ["bl_daly"]+eur_econ["bl_daly"]+sear_econ["bl_daly"]+wpr_econ["bl_daly"]
global_econ["s1_daly"]=afr_econ["s1_daly"]+amr_econ["s1_daly"]+emr_econ["s1_daly"]+eur_econ["s1_daly"]+sear_econ["s1_daly"]+wpr_econ["s1_daly"]
global_econ["s2_daly"]=afr_econ["s2_daly"]+amr_econ["s2_daly"]+emr_econ["s2_daly"]+eur_econ["s2_daly"]+sear_econ["s2_daly"]+wpr_econ["s2_daly"]
global_econ["s3_daly"]=afr_econ["s3_daly"]+amr_econ["s3_daly"]+emr_econ["s3_daly"]+eur_econ["s3_daly"]+sear_econ["s3_daly"]+wpr_econ["s3_daly"]

global_econ["cbl_dirc"]=afr_econ["cbl_dirc"]+amr_econ["cbl_dirc"]+emr_econ["cbl_dirc"]+eur_econ["cbl_dirc"]+sear_econ["cbl_dirc"]+wpr_econ["cbl_dirc"]
global_econ["cs1_dirc"]=afr_econ["cs1_dirc"]+amr_econ["cs1_dirc"]+emr_econ["cs1_dirc"]+eur_econ["cs1_dirc"]+sear_econ["cs1_dirc"]+wpr_econ["cs1_dirc"]
global_econ["cs2_dirc"]=afr_econ["cs2_dirc"]+amr_econ["cs2_dirc"]+emr_econ["cs2_dirc"]+eur_econ["cs2_dirc"]+sear_econ["cs2_dirc"]+wpr_econ["cs2_dirc"]
global_econ["cs3_dirc"]=afr_econ["cs3_dirc"]+amr_econ["cs3_dirc"]+emr_econ["cs3_dirc"]+eur_econ["cs3_dirc"]+sear_econ["cs3_dirc"]+wpr_econ["cs3_dirc"]

global_econ["bl_dirc"]=afr_econ["bl_dirc"]+amr_econ["bl_dirc"]+emr_econ["bl_dirc"]+eur_econ["bl_dirc"]+sear_econ["bl_dirc"]+wpr_econ["bl_dirc"]
global_econ["s1_dirc"]=afr_econ["s1_dirc"]+amr_econ["s1_dirc"]+emr_econ["s1_dirc"]+eur_econ["s1_dirc"]+sear_econ["s1_dirc"]+wpr_econ["s1_dirc"]
global_econ["s2_dirc"]=afr_econ["s2_dirc"]+amr_econ["s2_dirc"]+emr_econ["s2_dirc"]+eur_econ["s2_dirc"]+sear_econ["s2_dirc"]+wpr_econ["s2_dirc"]
global_econ["s3_dirc"]=afr_econ["s3_dirc"]+amr_econ["s3_dirc"]+emr_econ["s3_dirc"]+eur_econ["s3_dirc"]+sear_econ["s3_dirc"]+wpr_econ["s3_dirc"]

global_econ["cs1_icer"]=-((afr_econ["cbl_cumdc"]+amr_econ["cbl_cumdc"]+emr_econ["cbl_cumdc"]+eur_econ["cbl_cumdc"]+sear_econ["cbl_cumdc"]+wpr_econ["cbl_cumdc"])-(afr_econ["cs1_cumdc"]+amr_econ["cs1_cumdc"]+emr_econ["cs1_cumdc"]+eur_econ["cs1_cumdc"]+sear_econ["cs1_cumdc"]+wpr_econ["cs1_cumdc"]))/((afr_econ["cbl_cdal"]+amr_econ["cbl_cdal"]+emr_econ["cbl_cdal"]+eur_econ["cbl_cdal"]+sear_econ["cbl_cdal"]+wpr_econ["cbl_cdal"])-(afr_econ["cs1_cdal"]+amr_econ["cs1_cdal"]+emr_econ["cs1_cdal"]+eur_econ["cs1_cdal"]+sear_econ["cs1_cdal"]+wpr_econ["cs1_cdal"]))
global_econ["cs2_icer"]=-((afr_econ["cbl_cumdc"]+amr_econ["cbl_cumdc"]+emr_econ["cbl_cumdc"]+eur_econ["cbl_cumdc"]+sear_econ["cbl_cumdc"]+wpr_econ["cbl_cumdc"])-(afr_econ["cs2_cumdc"]+amr_econ["cs2_cumdc"]+emr_econ["cs2_cumdc"]+eur_econ["cs2_cumdc"]+sear_econ["cs2_cumdc"]+wpr_econ["cs2_cumdc"]))/((afr_econ["cbl_cdal"]+amr_econ["cbl_cdal"]+emr_econ["cbl_cdal"]+eur_econ["cbl_cdal"]+sear_econ["cbl_cdal"]+wpr_econ["cbl_cdal"])-(afr_econ["cs2_cdal"]+amr_econ["cs2_cdal"]+emr_econ["cs2_cdal"]+eur_econ["cs2_cdal"]+sear_econ["cs2_cdal"]+wpr_econ["cs2_cdal"]))
global_econ["cs3_icer"]=-((afr_econ["cbl_cumdc"]+amr_econ["cbl_cumdc"]+emr_econ["cbl_cumdc"]+eur_econ["cbl_cumdc"]+sear_econ["cbl_cumdc"]+wpr_econ["cbl_cumdc"])-(afr_econ["cs3_cumdc"]+amr_econ["cs3_cumdc"]+emr_econ["cs3_cumdc"]+eur_econ["cs3_cumdc"]+sear_econ["cs3_cumdc"]+wpr_econ["cs3_cumdc"]))/((afr_econ["cbl_cdal"]+amr_econ["cbl_cdal"]+emr_econ["cbl_cdal"]+eur_econ["cbl_cdal"]+sear_econ["cbl_cdal"]+wpr_econ["cbl_cdal"])-(afr_econ["cs3_cdal"]+amr_econ["cs3_cdal"]+emr_econ["cs3_cdal"]+eur_econ["cs3_cdal"]+sear_econ["cs3_cdal"]+wpr_econ["cs3_cdal"]))

global_econ["s1_icer"]=-((afr_econ["bl_cumdc"]+amr_econ["bl_cumdc"]+emr_econ["bl_cumdc"]+eur_econ["bl_cumdc"]+sear_econ["bl_cumdc"]+wpr_econ["bl_cumdc"])-(afr_econ["s1_cumdc"]+amr_econ["s1_cumdc"]+emr_econ["s1_cumdc"]+eur_econ["s1_cumdc"]+sear_econ["s1_cumdc"]+wpr_econ["s1_cumdc"]))/((afr_econ["bl_cdal"]+amr_econ["bl_cdal"]+emr_econ["bl_cdal"]+eur_econ["bl_cdal"]+sear_econ["bl_cdal"]+wpr_econ["bl_cdal"])-(afr_econ["s1_cdal"]+amr_econ["s1_cdal"]+emr_econ["s1_cdal"]+eur_econ["s1_cdal"]+sear_econ["s1_cdal"]+wpr_econ["s1_cdal"]))
global_econ["s2_icer"]=-((afr_econ["bl_cumdc"]+amr_econ["bl_cumdc"]+emr_econ["bl_cumdc"]+eur_econ["bl_cumdc"]+sear_econ["bl_cumdc"]+wpr_econ["bl_cumdc"])-(afr_econ["s2_cumdc"]+amr_econ["s2_cumdc"]+emr_econ["s2_cumdc"]+eur_econ["s2_cumdc"]+sear_econ["s2_cumdc"]+wpr_econ["s2_cumdc"]))/((afr_econ["bl_cdal"]+amr_econ["bl_cdal"]+emr_econ["bl_cdal"]+eur_econ["bl_cdal"]+sear_econ["bl_cdal"]+wpr_econ["bl_cdal"])-(afr_econ["s2_cdal"]+amr_econ["s2_cdal"]+emr_econ["s2_cdal"]+eur_econ["s2_cdal"]+sear_econ["s2_cdal"]+wpr_econ["s2_cdal"]))
global_econ["s3_icer"]=-((afr_econ["bl_cumdc"]+amr_econ["bl_cumdc"]+emr_econ["bl_cumdc"]+eur_econ["bl_cumdc"]+sear_econ["bl_cumdc"]+wpr_econ["bl_cumdc"])-(afr_econ["s3_cumdc"]+amr_econ["s3_cumdc"]+emr_econ["s3_cumdc"]+eur_econ["s3_cumdc"]+sear_econ["s3_cumdc"]+wpr_econ["s3_cumdc"]))/((afr_econ["bl_cdal"]+amr_econ["bl_cdal"]+emr_econ["bl_cdal"]+eur_econ["bl_cdal"]+sear_econ["bl_cdal"]+wpr_econ["bl_cdal"])-(afr_econ["s3_cdal"]+amr_econ["s3_cdal"]+emr_econ["s3_cdal"]+eur_econ["s3_cdal"]+sear_econ["s3_cdal"]+wpr_econ["s3_cdal"]))

global_econ["cs1_neb"]=afr_econ["cs1_neb"]+amr_econ["cs1_neb"]+emr_econ["cs1_neb"]+eur_econ["cs1_neb"]+sear_econ["cs1_neb"]+wpr_econ["cs1_neb"]
global_econ["cs2_neb"]=afr_econ["cs2_neb"]+amr_econ["cs2_neb"]+emr_econ["cs2_neb"]+eur_econ["cs2_neb"]+sear_econ["cs2_neb"]+wpr_econ["cs2_neb"]
global_econ["cs3_neb"]=afr_econ["cs3_neb"]+amr_econ["cs3_neb"]+emr_econ["cs3_neb"]+eur_econ["cs3_neb"]+sear_econ["cs3_neb"]+wpr_econ["cs3_neb"]

global_econ["s1_neb"]=afr_econ["s1_neb"]+amr_econ["s1_neb"]+emr_econ["s1_neb"]+eur_econ["s1_neb"]+sear_econ["s1_neb"]+wpr_econ["s1_neb"]
global_econ["s2_neb"]=afr_econ["s2_neb"]+amr_econ["s2_neb"]+emr_econ["s2_neb"]+eur_econ["s2_neb"]+sear_econ["s2_neb"]+wpr_econ["s2_neb"]
global_econ["s3_neb"]=afr_econ["s3_neb"]+amr_econ["s3_neb"]+emr_econ["s3_neb"]+eur_econ["s3_neb"]+sear_econ["s3_neb"]+wpr_econ["s3_neb"]

## FIX: sanitize global ICERs and NEBs (inf/nan from early-year 0/0 division or NaN propagation)
for _key in ["cs1_icer","cs2_icer","cs3_icer","s1_icer","s2_icer","s3_icer",
             "cs1_neb","cs2_neb","cs3_neb","s1_neb","s2_neb","s3_neb"]:
    global_econ[_key] = np.nan_to_num(global_econ[_key], nan=0.0, posinf=0.0, neginf=0.0)

econ_plots (global_econ, "global")

#%% Table 2 Creation

def summary_stats_median(bl, s1, s2, s3, econ,lb,ub):
    """Returns summary stats for each region, can be 
    collated post-hoc for table 2"""
    
    import numpy as np
    
    error=[50,lb,ub] #median and bounds
    bl_tab_vars=["chb_inc", "hcc_inc", "mort", "bl_daly", "bl_intc", "bl_medc", "bl_prod"]
    s1_tab_vars=["chb_inc", "hcc_inc", "mort", "s1_daly", "s1_intc", "s1_medc", "s1_prod"]
    s2_tab_vars=["chb_inc", "hcc_inc", "mort", "s2_daly", "s2_intc", "s2_medc", "s2_prod"]
    s3_tab_vars=["chb_inc", "hcc_inc", "mort", "s3_daly", "s3_intc", "s3_medc", "s3_prod"]
    
    bl_sum=np.zeros((8,3))
    s1_sum=np.zeros((8,3))
    s2_sum=np.zeros((8,3))
    s3_sum=np.zeros((8,3))
    
    
    for idx,val in enumerate(bl_tab_vars):
        for i, err in enumerate(error):
            if idx==0:      #change this when re-running the model with CHB incidence added!
                bl_sum[idx,i]=np.percentile(np.sum(bl[val][36:110,:],axis=0),err)
            elif idx >0 and idx <3:
                bl_sum[idx,i]=np.percentile(np.sum(bl[val][36:110,:],axis=0),err)
            elif idx>=3:
                bl_sum[idx,i]=np.percentile(np.sum(econ[val][36:110,:],axis=0),err)
                
    for idx,val in enumerate(s1_tab_vars):
        for i, err in enumerate(error):
            if idx==0:      #change this when re-running the model with CHB incidence added!
                s1_sum[idx,i]=np.percentile(np.sum(s1[val][36:110,:],axis=0),err)
            elif idx >0 and idx <3:
                s1_sum[idx,i]=np.percentile(np.sum(s1[val][36:110,:],axis=0),err)
            elif idx>=3:
                s1_sum[idx,i]=np.percentile(np.sum(econ[val][36:110,:],axis=0),err)
    
    for idx,val in enumerate(s2_tab_vars):
        for i, err in enumerate(error):
            if idx==0:      #change this when re-running the model with CHB incidence added!
                s2_sum[idx,i]=np.percentile(np.sum(s2[val][36:110,:],axis=0),err)
            elif idx >0 and idx <3:
                s2_sum[idx,i]=np.percentile(np.sum(s2[val][36:110,:],axis=0),err)
            elif idx>=3:
                s2_sum[idx,i]=np.percentile(np.sum(econ[val][36:110,:],axis=0),err)
                
    for idx,val in enumerate(s3_tab_vars):
        for i, err in enumerate(error):
            if idx==0:      #change this when re-running the model with CHB incidence added!
                s3_sum[idx,i]=np.percentile(np.sum(s3[val][36:110,:],axis=0),err)
            elif idx >0 and idx <3:
                s3_sum[idx,i]=np.percentile(np.sum(s3[val][36:110,:],axis=0),err)
            elif idx>=3:
                s3_sum[idx,i]=np.percentile(np.sum(econ[val][36:110,:],axis=0),err)
    
    #Add years (with bounds) net economic benefit is reached
    s1_neb_med=np.zeros((110,2))
    s1_neb_med[:,0]=np.arange(1990, 2100, 1)
    s1_neb_med[:,1]=np.percentile(econ["s1_neb"],50, axis=1)
    year=[]
    for idx,val in enumerate(s1_neb_med[:,1]):
        if val >0 and s1_neb_med[idx,0] >= 2027:
            year.append(s1_neb_med[idx,0])
    s1_sum[7,0]=safe_min(year)
    
    s1_neb_lb=np.zeros((110,2))
    s1_neb_lb[:,0]=np.arange(1990, 2100, 1)
    s1_neb_lb[:,1]=np.percentile(econ["s1_neb"],97.5, axis=1)
    year=[]
    for idx,val in enumerate(s1_neb_lb[:,1]):
        if val >0 and s1_neb_lb[idx,0] >= 2027:
            year.append(s1_neb_lb[idx,0])
    s1_sum[7,1]=safe_min(year)
    
    s1_neb_ub=np.zeros((110,2))
    s1_neb_ub[:,0]=np.arange(1990, 2100, 1)
    s1_neb_ub[:,1]=np.percentile(econ["s1_neb"],2.5, axis=1)
    year=[]
    for idx,val in enumerate(s1_neb_ub[:,1]):
        if val >0 and s1_neb_ub[idx,0] >= 2027:
            year.append(s1_neb_ub[idx,0])
    s1_sum[7,2]=safe_min(year)
    
    
    s2_neb_med=np.zeros((110,2))
    s2_neb_med[:,0]=np.arange(1990, 2100, 1)
    s2_neb_med[:,1]=np.percentile(econ["s2_neb"],50, axis=1)
    year=[]
    for idx,val in enumerate(s2_neb_med[:,1]):
        if val >0 and s2_neb_med[idx,0] >= 2027:
            year.append(s2_neb_med[idx,0])
    s2_sum[7,0]=safe_min(year)
    
    s2_neb_lb=np.zeros((110,2))
    s2_neb_lb[:,0]=np.arange(1990, 2100, 1)
    s2_neb_lb[:,1]=np.percentile(econ["s2_neb"],97.5, axis=1)
    year=[]
    for idx,val in enumerate(s2_neb_lb[:,1]):
        if val >0 and s2_neb_lb[idx,0] >= 2027:
            year.append(s2_neb_lb[idx,0])
    s2_sum[7,1]=safe_min(year)
    
    s2_neb_ub=np.zeros((110,2))
    s2_neb_ub[:,0]=np.arange(1990, 2100, 1)
    s2_neb_ub[:,1]=np.percentile(econ["s2_neb"],2.5, axis=1)
    year=[]
    for idx,val in enumerate(s2_neb_ub[:,1]):
        if val >0 and s2_neb_ub[idx,0] >= 2027:
            year.append(s2_neb_ub[idx,0])
    s2_sum[7,2]=safe_min(year)
    
    
    s3_neb_med=np.zeros((110,2))
    s3_neb_med[:,0]=np.arange(1990, 2100, 1)
    s3_neb_med[:,1]=np.percentile(econ["s3_neb"],50, axis=1)
    year=[]
    for idx,val in enumerate(s3_neb_med[:,1]):
        if val >0 and s3_neb_med[idx,0] >= 2027:
            year.append(s3_neb_med[idx,0])
    s3_sum[7,0]=safe_min(year)
    
    
    s3_neb_lb=np.zeros((110,2))
    s3_neb_lb[:,0]=np.arange(1990, 2100, 1)
    s3_neb_lb[:,1]=np.percentile(econ["s3_neb"],97.5, axis=1)
    year=[]
    for idx,val in enumerate(s3_neb_lb[:,1]):
        if val >0 and s3_neb_lb[idx,0] >= 2027:
            year.append(s3_neb_lb[idx,0])
    s3_sum[7,1]=safe_min(year)
    
    s3_neb_ub=np.zeros((110,2))
    s3_neb_ub[:,0]=np.arange(1990, 2100, 1)
    s3_neb_ub[:,1]=np.percentile(econ["s3_neb"],2.5, axis=1)
    year=[]
    for idx,val in enumerate(s3_neb_ub[:,1]):
        if val >0 and s3_neb_ub[idx,0] >= 2027:
            year.append(s3_neb_ub[idx,0])
    s3_sum[7,2]=safe_min(year)
    
    t2_sum=np.concatenate([bl_sum,s1_sum, s2_sum, s3_sum], axis=1)
    
    return t2_sum

tab_2=np.zeros((7,8,12))
tab_2[0,:,:]=summary_stats_median(afr_bl_runs, afr_s1_runs, afr_s2_runs, afr_s3_runs, afr_econ,2.5,97.5)
tab_2[1,:,:]=summary_stats_median(amr_bl_runs, amr_s1_runs, amr_s2_runs, amr_s3_runs, amr_econ,2.5,97.5)
tab_2[2,:,:]=summary_stats_median(emr_bl_runs, emr_s1_runs, emr_s2_runs, emr_s3_runs, emr_econ,2.5,97.5)
tab_2[3,:,:]=summary_stats_median(eur_bl_runs, eur_s1_runs, eur_s2_runs, eur_s3_runs, eur_econ,2.5,97.5)
tab_2[4,:,:]=summary_stats_median(sear_bl_runs, sear_s1_runs, sear_s2_runs, sear_s3_runs, sear_econ,2.5,97.5)
tab_2[5,:,:]=summary_stats_median(wpr_bl_runs, wpr_s1_runs, wpr_s2_runs, wpr_s3_runs, wpr_econ,2.5,97.5)
tab_2[6,:,:]=summary_stats_median(global_bl_runs, global_s1_runs, global_s2_runs, global_s3_runs, global_econ,2.5,97.5)

##Replace medians with central estimates

## AFR
year=np.arange(1990, 2100, 1)

tab_2[0,0,0]=np.sum(afr_bl_cent["chb_inc"][36:110])
tab_2[0,1,0]=np.sum(afr_bl_cent["hcc_inc"][36:110])
tab_2[0,2,0]=np.sum(afr_bl_cent["mort"][36:110])
tab_2[0,3,0]=np.sum(afr_econ["cbl_daly"][36:110])
tab_2[0,4,0]=np.sum(afr_econ["cbl_intc"][36:110])
tab_2[0,5,0]=np.sum(afr_econ["cbl_medc"][36:110])
tab_2[0,6,0]=np.sum(afr_econ["cbl_prod"][36:110])

tab_2[0,0,3]=np.sum(afr_s1_cent["chb_inc"][36:110])
tab_2[0,1,3]=np.sum(afr_s1_cent["hcc_inc"][36:110])
tab_2[0,2,3]=np.sum(afr_s1_cent["mort"][36:110])
tab_2[0,3,3]=np.sum(afr_econ["cs1_daly"][36:110])
tab_2[0,4,3]=np.sum(afr_econ["cs1_intc"][36:110])
tab_2[0,5,3]=np.sum(afr_econ["cs1_medc"][36:110])
tab_2[0,6,3]=np.sum(afr_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(afr_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[0,7,3]=year[yneb_i]

tab_2[0,0,6]=np.sum(afr_s2_cent["chb_inc"][36:110])
tab_2[0,1,6]=np.sum(afr_s2_cent["hcc_inc"][36:110])
tab_2[0,2,6]=np.sum(afr_s2_cent["mort"][36:110])
tab_2[0,3,6]=np.sum(afr_econ["cs2_daly"][36:110])
tab_2[0,4,6]=np.sum(afr_econ["cs2_intc"][36:110])
tab_2[0,5,6]=np.sum(afr_econ["cs2_medc"][36:110])
tab_2[0,6,6]=np.sum(afr_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(afr_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[0,7,6]=year[yneb_i]

tab_2[0,0,9]=np.sum(afr_s3_cent["chb_inc"][36:110])
tab_2[0,1,9]=np.sum(afr_s3_cent["hcc_inc"][36:110])
tab_2[0,2,9]=np.sum(afr_s3_cent["mort"][36:110])
tab_2[0,3,9]=np.sum(afr_econ["cs3_daly"][36:110])
tab_2[0,4,9]=np.sum(afr_econ["cs3_intc"][36:110])
tab_2[0,5,9]=np.sum(afr_econ["cs3_medc"][36:110])
tab_2[0,6,9]=np.sum(afr_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(afr_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[0,7,9]=year[yneb_i]


## AMR 
tab_2[1,0,0]=np.sum(amr_bl_cent["chb_inc"][36:110])
tab_2[1,1,0]=np.sum(amr_bl_cent["hcc_inc"][36:110])
tab_2[1,2,0]=np.sum(amr_bl_cent["mort"][36:110])
tab_2[1,3,0]=np.sum(amr_econ["cbl_daly"][36:110])
tab_2[1,4,0]=np.sum(amr_econ["cbl_intc"][36:110])
tab_2[1,5,0]=np.sum(amr_econ["cbl_medc"][36:110])
tab_2[1,6,0]=np.sum(amr_econ["cbl_prod"][36:110])

tab_2[1,0,3]=np.sum(amr_s1_cent["chb_inc"][36:110])
tab_2[1,1,3]=np.sum(amr_s1_cent["hcc_inc"][36:110])
tab_2[1,2,3]=np.sum(amr_s1_cent["mort"][36:110])
tab_2[1,3,3]=np.sum(amr_econ["cs1_daly"][36:110])
tab_2[1,4,3]=np.sum(amr_econ["cs1_intc"][36:110])
tab_2[1,5,3]=np.sum(amr_econ["cs1_medc"][36:110])
tab_2[1,6,3]=np.sum(amr_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(amr_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[1,7,3]=year[yneb_i]

tab_2[1,0,6]=np.sum(amr_s2_cent["chb_inc"][36:110])
tab_2[1,1,6]=np.sum(amr_s2_cent["hcc_inc"][36:110])
tab_2[1,2,6]=np.sum(amr_s2_cent["mort"][36:110])
tab_2[1,3,6]=np.sum(amr_econ["cs2_daly"][36:110])
tab_2[1,4,6]=np.sum(amr_econ["cs2_intc"][36:110])
tab_2[1,5,6]=np.sum(amr_econ["cs2_medc"][36:110])
tab_2[1,6,6]=np.sum(amr_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(amr_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[1,7,6]=year[yneb_i]

tab_2[1,0,9]=np.sum(amr_s3_cent["chb_inc"][36:110])
tab_2[1,1,9]=np.sum(amr_s3_cent["hcc_inc"][36:110])
tab_2[1,2,9]=np.sum(amr_s3_cent["mort"][36:110])
tab_2[1,3,9]=np.sum(amr_econ["cs3_daly"][36:110])
tab_2[1,4,9]=np.sum(amr_econ["cs3_intc"][36:110])
tab_2[1,5,9]=np.sum(amr_econ["cs3_medc"][36:110])
tab_2[1,6,9]=np.sum(amr_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(amr_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[1,7,9]=year[yneb_i]

## EMR
tab_2[2,0,0]=np.sum(emr_bl_cent["chb_inc"][36:110])
tab_2[2,1,0]=np.sum(emr_bl_cent["hcc_inc"][36:110])
tab_2[2,2,0]=np.sum(emr_bl_cent["mort"][36:110])
tab_2[2,3,0]=np.sum(emr_econ["cbl_daly"][36:110])
tab_2[2,4,0]=np.sum(emr_econ["cbl_intc"][36:110])
tab_2[2,5,0]=np.sum(emr_econ["cbl_medc"][36:110])
tab_2[2,6,0]=np.sum(emr_econ["cbl_prod"][36:110])

tab_2[2,0,3]=np.sum(emr_s1_cent["chb_inc"][36:110])
tab_2[2,1,3]=np.sum(emr_s1_cent["hcc_inc"][36:110])
tab_2[2,2,3]=np.sum(emr_s1_cent["mort"][36:110])
tab_2[2,3,3]=np.sum(emr_econ["cs1_daly"][36:110])
tab_2[2,4,3]=np.sum(emr_econ["cs1_intc"][36:110])
tab_2[2,5,3]=np.sum(emr_econ["cs1_medc"][36:110])
tab_2[2,6,3]=np.sum(emr_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(emr_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[2,7,3]=year[yneb_i]

tab_2[2,0,6]=np.sum(emr_s2_cent["chb_inc"][36:110])
tab_2[2,1,6]=np.sum(emr_s2_cent["hcc_inc"][36:110])
tab_2[2,2,6]=np.sum(emr_s2_cent["mort"][36:110])
tab_2[2,3,6]=np.sum(emr_econ["cs2_daly"][36:110])
tab_2[2,4,6]=np.sum(emr_econ["cs2_intc"][36:110])
tab_2[2,5,6]=np.sum(emr_econ["cs2_medc"][36:110])
tab_2[2,6,6]=np.sum(emr_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(emr_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[2,7,6]=year[yneb_i]

tab_2[2,0,9]=np.sum(emr_s3_cent["chb_inc"][36:110])
tab_2[2,1,9]=np.sum(emr_s3_cent["hcc_inc"][36:110])
tab_2[2,2,9]=np.sum(emr_s3_cent["mort"][36:110])
tab_2[2,3,9]=np.sum(emr_econ["cs3_daly"][36:110])
tab_2[2,4,9]=np.sum(emr_econ["cs3_intc"][36:110])
tab_2[2,5,9]=np.sum(emr_econ["cs3_medc"][36:110])
tab_2[2,6,9]=np.sum(emr_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(emr_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[2,7,9]=year[yneb_i]

## EUR 
tab_2[3,1,0]=np.sum(eur_bl_cent["hcc_inc"][36:110])
tab_2[3,2,0]=np.sum(eur_bl_cent["mort"][36:110])
tab_2[3,3,0]=np.sum(eur_econ["cbl_daly"][36:110])
tab_2[3,4,0]=np.sum(eur_econ["cbl_intc"][36:110])
tab_2[3,5,0]=np.sum(eur_econ["cbl_medc"][36:110])
tab_2[3,6,0]=np.sum(eur_econ["cbl_prod"][36:110])

tab_2[3,0,3]=np.sum(eur_s1_cent["chb_inc"][36:110])
tab_2[3,1,3]=np.sum(eur_s1_cent["hcc_inc"][36:110])
tab_2[3,2,3]=np.sum(eur_s1_cent["mort"][36:110])
tab_2[3,3,3]=np.sum(eur_econ["cs1_daly"][36:110])
tab_2[3,4,3]=np.sum(eur_econ["cs1_intc"][36:110])
tab_2[3,5,3]=np.sum(eur_econ["cs1_medc"][36:110])
tab_2[3,6,3]=np.sum(eur_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(eur_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[3,7,3]=year[yneb_i]

tab_2[3,0,6]=np.sum(eur_s2_cent["chb_inc"][36:110])
tab_2[3,1,6]=np.sum(eur_s2_cent["hcc_inc"][36:110])
tab_2[3,2,6]=np.sum(eur_s2_cent["mort"][36:110])
tab_2[3,3,6]=np.sum(eur_econ["cs2_daly"][36:110])
tab_2[3,4,6]=np.sum(eur_econ["cs2_intc"][36:110])
tab_2[3,5,6]=np.sum(eur_econ["cs2_medc"][36:110])
tab_2[3,6,6]=np.sum(eur_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(eur_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[3,7,6]=year[yneb_i]

tab_2[3,0,9]=np.sum(eur_s3_cent["chb_inc"][36:110])
tab_2[3,1,9]=np.sum(eur_s3_cent["hcc_inc"][36:110])
tab_2[3,2,9]=np.sum(eur_s3_cent["mort"][36:110])
tab_2[3,3,9]=np.sum(eur_econ["cs3_daly"][36:110])
tab_2[3,4,9]=np.sum(eur_econ["cs3_intc"][36:110])
tab_2[3,5,9]=np.sum(eur_econ["cs3_medc"][36:110])
tab_2[3,6,9]=np.sum(eur_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(eur_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[3,7,9]=year[yneb_i]

## SEAR
tab_2[4,1,0]=np.sum(sear_bl_cent["hcc_inc"][36:110])
tab_2[4,2,0]=np.sum(sear_bl_cent["mort"][36:110])
tab_2[4,3,0]=np.sum(sear_econ["cbl_daly"][36:110])
tab_2[4,4,0]=np.sum(sear_econ["cbl_intc"][36:110])
tab_2[4,5,0]=np.sum(sear_econ["cbl_medc"][36:110])
tab_2[4,6,0]=np.sum(sear_econ["cbl_prod"][36:110])

tab_2[4,0,3]=np.sum(sear_s1_cent["chb_inc"][36:110])
tab_2[4,1,3]=np.sum(sear_s1_cent["hcc_inc"][36:110])
tab_2[4,2,3]=np.sum(sear_s1_cent["mort"][36:110])
tab_2[4,3,3]=np.sum(sear_econ["cs1_daly"][36:110])
tab_2[4,4,3]=np.sum(sear_econ["cs1_intc"][36:110])
tab_2[4,5,3]=np.sum(sear_econ["cs1_medc"][36:110])
tab_2[4,6,3]=np.sum(sear_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(sear_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[4,7,3]=year[yneb_i]

tab_2[4,0,6]=np.sum(sear_s2_cent["chb_inc"][36:110])
tab_2[4,1,6]=np.sum(sear_s2_cent["hcc_inc"][36:110])
tab_2[4,2,6]=np.sum(sear_s2_cent["mort"][36:110])
tab_2[4,3,6]=np.sum(sear_econ["cs2_daly"][36:110])
tab_2[4,4,6]=np.sum(sear_econ["cs2_intc"][36:110])
tab_2[4,5,6]=np.sum(sear_econ["cs2_medc"][36:110])
tab_2[4,6,6]=np.sum(sear_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(sear_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[4,7,6]=year[yneb_i]

tab_2[4,0,9]=np.sum(sear_s3_cent["chb_inc"][36:110])
tab_2[4,1,9]=np.sum(sear_s3_cent["hcc_inc"][36:110])
tab_2[4,2,9]=np.sum(sear_s3_cent["mort"][36:110])
tab_2[4,3,9]=np.sum(sear_econ["cs3_daly"][36:110])
tab_2[4,4,9]=np.sum(sear_econ["cs3_intc"][36:110])
tab_2[4,5,9]=np.sum(sear_econ["cs3_medc"][36:110])
tab_2[4,6,9]=np.sum(sear_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(sear_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[4,7,9]=year[yneb_i]

## WPR 
tab_2[5,1,0]=np.sum(wpr_bl_cent["hcc_inc"][36:110])
tab_2[5,2,0]=np.sum(wpr_bl_cent["mort"][36:110])
tab_2[5,3,0]=np.sum(wpr_econ["cbl_daly"][36:110])
tab_2[5,4,0]=np.sum(wpr_econ["cbl_intc"][36:110])
tab_2[5,5,0]=np.sum(wpr_econ["cbl_medc"][36:110])
tab_2[5,6,0]=np.sum(wpr_econ["cbl_prod"][36:110])

tab_2[5,0,3]=np.sum(wpr_s1_cent["chb_inc"][36:110])
tab_2[5,1,3]=np.sum(wpr_s1_cent["hcc_inc"][36:110])
tab_2[5,2,3]=np.sum(wpr_s1_cent["mort"][36:110])
tab_2[5,3,3]=np.sum(wpr_econ["cs1_daly"][36:110])
tab_2[5,4,3]=np.sum(wpr_econ["cs1_intc"][36:110])
tab_2[5,5,3]=np.sum(wpr_econ["cs1_medc"][36:110])
tab_2[5,6,3]=np.sum(wpr_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(wpr_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[5,7,3]=year[yneb_i]

tab_2[5,0,6]=np.sum(wpr_s2_cent["chb_inc"][36:110])
tab_2[5,1,6]=np.sum(wpr_s2_cent["hcc_inc"][36:110])
tab_2[5,2,6]=np.sum(wpr_s2_cent["mort"][36:110])
tab_2[5,3,6]=np.sum(wpr_econ["cs2_daly"][36:110])
tab_2[5,4,6]=np.sum(wpr_econ["cs2_intc"][36:110])
tab_2[5,5,6]=np.sum(wpr_econ["cs2_medc"][36:110])
tab_2[5,6,6]=np.sum(wpr_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(wpr_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[5,7,6]=year[yneb_i]

tab_2[5,0,9]=np.sum(wpr_s3_cent["chb_inc"][36:110])
tab_2[5,1,9]=np.sum(wpr_s3_cent["hcc_inc"][36:110])
tab_2[5,2,9]=np.sum(wpr_s3_cent["mort"][36:110])
tab_2[5,3,9]=np.sum(wpr_econ["cs3_daly"][36:110])
tab_2[5,4,9]=np.sum(wpr_econ["cs3_intc"][36:110])
tab_2[5,5,9]=np.sum(wpr_econ["cs3_medc"][36:110])
tab_2[5,6,9]=np.sum(wpr_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(wpr_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[5,7,9]=year[yneb_i]

## Global 
tab_2[6,1,0]=np.sum(global_bl_cent["hcc_inc"][36:110])
tab_2[6,2,0]=np.sum(global_bl_cent["mort"][36:110])
tab_2[6,3,0]=np.sum(global_econ["cbl_daly"][36:110])
tab_2[6,4,0]=np.sum(global_econ["cbl_intc"][36:110])
tab_2[6,5,0]=np.sum(global_econ["cbl_medc"][36:110])
tab_2[6,6,0]=np.sum(global_econ["cbl_prod"][36:110])

tab_2[6,0,3]=np.sum(global_s1_cent["chb_inc"][36:110])
tab_2[6,1,3]=np.sum(global_s1_cent["hcc_inc"][36:110])
tab_2[6,2,3]=np.sum(global_s1_cent["mort"][36:110])
tab_2[6,3,3]=np.sum(global_econ["cs1_daly"][36:110])
tab_2[6,4,3]=np.sum(global_econ["cs1_intc"][36:110])
tab_2[6,5,3]=np.sum(global_econ["cs1_medc"][36:110])
tab_2[6,6,3]=np.sum(global_econ["cs1_prod"][36:110])
year_idx=[]
for idx,val in enumerate(global_econ["cs1_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[6,7,3]=year[yneb_i]

tab_2[6,0,6]=np.sum(global_s2_cent["chb_inc"][36:110])
tab_2[6,1,6]=np.sum(global_s2_cent["hcc_inc"][36:110])
tab_2[6,2,6]=np.sum(global_s2_cent["mort"][36:110])
tab_2[6,3,6]=np.sum(global_econ["cs2_daly"][36:110])
tab_2[6,4,6]=np.sum(global_econ["cs2_intc"][36:110])
tab_2[6,5,6]=np.sum(global_econ["cs2_medc"][36:110])
tab_2[6,6,6]=np.sum(global_econ["cs2_prod"][36:110])
year_idx=[]
for idx,val in enumerate(global_econ["cs2_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[6,7,6]=year[yneb_i]

tab_2[6,0,9]=np.sum(global_s3_cent["chb_inc"][36:110])
tab_2[6,1,9]=np.sum(global_s3_cent["hcc_inc"][36:110])
tab_2[6,2,9]=np.sum(global_s3_cent["mort"][36:110])
tab_2[6,3,9]=np.sum(global_econ["cs3_daly"][36:110])
tab_2[6,4,9]=np.sum(global_econ["cs3_intc"][36:110])
tab_2[6,5,9]=np.sum(global_econ["cs3_medc"][36:110])
tab_2[6,6,9]=np.sum(global_econ["cs3_prod"][36:110])
year_idx=[]
for idx,val in enumerate(global_econ["cs3_neb"]):
    if val>0 and year[idx] >= 2027:
        year_idx.append(idx)
yneb_i=safe_min(year_idx)
tab_2[6,7,9]=year[yneb_i]


## Save Table 2 as a spreadsheet
writer=pd.ExcelWriter("Table 2_fc.xlsx", engine="xlsxwriter")

region=["AFR", "AMR", "EMR", "EUR", "SEAR", "WPR", "Global"]

for i in range(len(region)):
    df=pd.DataFrame(tab_2[i,:,:])
    df.to_excel(writer, sheet_name=region[i])

writer.close()

#%% Sensitivity Analysis 

## Note: This code block completes all presented sensitivity analyses (takes 20-30 mins of computational time)

## MTCT worst case
mtct_wc=[ #HBeAg negative HepB-BD effectiveness
         {"0-4M_sve":["sag_ve","0-4M",1990, 2021, 2022,0.95, 0.95, 0.86]},
         {"0-4F_sve":["sag_ve","0-4F",1990, 2021, 2022,0.95, 0.95, 0.86]},
         #HBeAg positive HepB-BD effectiveness
         {"0-4M_eve":["eag_ve","0-4M",1990, 2021, 2022,0.74, 0.74, 0.65]},
         {"0-4F_eve":["eag_ve","0-4F",1990, 2021, 2022,0.74, 0.74, 0.65]},
         #HBeAg positive HepB-BD+mAV effectiveness
         {"0-4M_mve":["mav_ve","0-4M",1990, 2021, 2022,0.98, 0.98, 0.97]},
         {"0-4F_mve":["mav_ve","0-4F",1990, 2021, 2022,0.98, 0.98, 0.97]},
         #HVL among HBeAg positive pregnancies
         {"0-4M_evl":["eag_hvl","0-4M",1990, 2021, 2022,0.91, 0.91, 0.96]},
         {"0-4F_evl":["eag_hvl","0-4F",1990, 2021, 2022,0.91, 0.91, 0.96]},
         #HVL among HBeAg negative pregnancies
         {"0-4M_svl":["sag_hvl","0-4M",1990, 2021, 2022,0.11, 0.11, 0.16]},
         {"0-4F_svl":["sag_hvl","0-4F",1990, 2021, 2022,0.11, 0.11, 0.16]},
         #HVL transmission risk
         {"0-4M_rsk":["hvl_trisk","0-4M",1990, 2021, 2022,0.85, 0.85, 1]},
         {"0-4F_rsk":["hvl_trisk","0-4F",1990, 2021, 2022,0.85, 0.85, 1]},
         #Risk of CHB following MTCT
         {"0-4M_cip":["ci_p","0-4M",1990, 2021, 2022,0.885, 0.885, 0.93]},
         {"0-4F_cip":["ci_p","0-4F",1990, 2021, 2022,0.885, 0.885, 0.93]}]

## MTCT best case
mtct_bc=[ #HBeAg negative HepB-BD effectiveness
         {"0-4M_sve":["sag_ve","0-4M",1990, 2021, 2022,0.95, 0.95, 0.99]},
         {"0-4F_sve":["sag_ve","0-4F",1990, 2021, 2022,0.95, 0.95, 0.99]},
         #HBeAg positive HepB-BD effectiveness
         {"0-4M_eve":["eag_ve","0-4M",1990, 2021, 2022,0.74, 0.74, 0.84]},
         {"0-4F_eve":["eag_ve","0-4F",1990, 2021, 2022,0.74, 0.74, 0.84]},
         #HBeAg positive HepB-BD+mAV effectiveness
         {"0-4M_mve":["mav_ve","0-4M",1990, 2021, 2022,0.98, 0.98, 0.99]},
         {"0-4F_mve":["mav_ve","0-4F",1990, 2021, 2022,0.98, 0.98, 0.99]},
         #HVL among HBeAg positive pregnancies
         {"0-4M_evl":["eag_hvl","0-4M",1990, 2021, 2022,0.91, 0.91, 0.84]},
         {"0-4F_evl":["eag_hvl","0-4F",1990, 2021, 2022,0.91, 0.91, 0.84]},
         #HVL among HBeAg negative pregnancies
         {"0-4M_svl":["sag_hvl","0-4M",1990, 2021, 2022,0.11, 0.11, 0.05]},
         {"0-4F_svl":["sag_hvl","0-4F",1990, 2021, 2022,0.11, 0.11, 0.05]},
         #HVL transmission risk
         {"0-4M_rsk":["hvl_trisk","0-4M",1990, 2021, 2022,0.85, 0.85, 0.7]},
         {"0-4F_rsk":["hvl_trisk","0-4F",1990, 2021, 2022,0.85, 0.85, 0.7]},
         #Risk of CHB following MTCT
         {"0-4M_cip":["ci_p","0-4M",1990, 2021, 2022,0.885, 0.885, 0.84]},
         {"0-4F_cip":["ci_p","0-4F",1990, 2021, 2022,0.885, 0.885, 0.84]}]

## Mortality worst case
mort_wc=[ # Acute mortality rate
         {"0-4M_acu":["m_acu", "0-4M", 1990, 2021, 2022, 0.00073, 0.00073, 0.00075]},
         {"0-4F_acu":["m_acu", "0-4F", 1990, 2021, 2022, 0.00073, 0.00073, 0.00075]},
         {"5-14M_acu":["m_acu", "5-14M", 1990, 2021, 2022, 0.0011, 0.0011, 0.0015]},
         {"5-14F_acu":["m_acu", "5-14F", 1990, 2021, 2022,0.0011, 0.0011, 0.0015]},
         {"15-49M_acu":["m_acu", "15-49M", 1990, 2021, 2022,0.0026, 0.0026, 0.00375]},
         {"15-49F_acu":["m_acu", "15-49F", 1990, 2021, 2022,0.0026, 0.0026, 0.00375]},
         {"50-69M_acu":["m_acu", "50-69M", 1990, 2021, 2022,0.0036, 0.0036, 0.00375]},
         {"50-69F_acu":["m_acu", "50-69F", 1990, 2021, 2022,0.0036, 0.0036, 0.00375]},
         {"70+M_acu":["m_acu", "70+M", 1990, 2021, 2022,0.0036, 0.0036, 0.00375]},
         {"70+F_acu":["m_acu", "70+F", 1990, 2021, 2022,0.0036, 0.0036, 0.00375]},
         # Decompensated cirrhosis mortality rate
         {"0-4M_dc":["m_dc", "0-4M", 1990, 2021, 2022, 0.15, 0.15, 0.2]},
         {"0-4F_dc":["m_dc", "0-4F", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"5-14M_dc":["m_dc", "5-14M", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"5-14F_dc":["m_dc", "5-14F", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"15-49M_dc":["m_dc", "15-49M", 1990, 2021, 2022, 0.15, 0.15, 0.2]},
         {"15-49F_dc":["m_dc", "15-49F", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"50-69M_dc":["m_dc", "50-69M", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"50-69F_dc":["m_dc", "50-69F", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"70+M_dc":["m_dc", "70+M", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         {"70+F_dc":["m_dc", "70+F", 1990, 2021, 2022,0.15, 0.15, 0.2]},
         # Hepatocellular carcinoma mortality rate
         {"5-14M_hcc":["m_hcc", "5-14M", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"5-14F_hcc":["m_hcc", "5-14F", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"15-49M_hcc":["m_hcc", "15-49M", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"15-49F_hcc":["m_hcc", "15-49F", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"50-69M_hcc":["m_hcc", "50-69M", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"50-69F_hcc":["m_hcc", "50-69F", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"70+M_hcc":["m_hcc", "70+M", 1990, 2021, 2022,0.313, 0.313, 0.545]},
         {"70+F_hcc":["m_hcc", "70+F", 1990, 2021, 2022,0.313, 0.313, 0.545]}]


## Mortalility best case
mort_bc=[ # Acute mortality rate
         {"0-4M_acu":["m_acu", "0-4M", 1990, 2021, 2022, 0.00073, 0.00073, 0.0007]},
         {"0-4F_acu":["m_acu", "0-4F", 1990, 2021, 2022, 0.00073, 0.00073, 0.0007]},
         {"5-14M_acu":["m_acu", "5-14M", 1990, 2021, 2022, 0.0011, 0.0011, 0.0007]},
         {"5-14F_acu":["m_acu", "5-14F", 1990, 2021, 2022,0.0011, 0.0011, 0.0007]},
         {"15-49M_acu":["m_acu", "15-49M", 1990, 2021, 2022,0.0026, 0.0026, 0.0007]},
         {"15-49F_acu":["m_acu", "15-49F", 1990, 2021, 2022,0.0026, 0.0026, 0.0007]},
         {"50-69M_acu":["m_acu", "50-69M", 1990, 2021, 2022,0.0036, 0.0036, 0.003]},
         {"50-69F_acu":["m_acu", "50-69F", 1990, 2021, 2022,0.0036, 0.0036, 0.003]},
         {"70+M_acu":["m_acu", "70+M", 1990, 2021, 2022,0.0036, 0.0036, 0.003]},
         {"70+F_acu":["m_acu", "70+F", 1990, 2021, 2022,0.0036, 0.0036, 0.003]},
         # Decompensated cirrhosis mortality rate
         {"0-4M_dc":["m_dc", "0-4M", 1990, 2021, 2022, 0.15, 0.15, 0.099]},
         {"0-4F_dc":["m_dc", "0-4F", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"5-14M_dc":["m_dc", "5-14M", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"5-14F_dc":["m_dc", "5-14F", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"15-49M_dc":["m_dc", "15-49M", 1990, 2021, 2022, 0.15, 0.15, 0.099]},
         {"15-49F_dc":["m_dc", "15-49F", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"50-69M_dc":["m_dc", "50-69M", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"50-69F_dc":["m_dc", "50-69F", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"70+M_dc":["m_dc", "70+M", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         {"70+F_dc":["m_dc", "70+F", 1990, 2021, 2022,0.15, 0.15, 0.099]},
         # Hepatocellular carcinoma mortality rate
         {"5-14M_hcc":["m_hcc", "5-14M", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"5-14F_hcc":["m_hcc", "5-14F", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"15-49M_hcc":["m_hcc", "15-49M", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"15-49F_hcc":["m_hcc", "15-49F", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"50-69M_hcc":["m_hcc", "50-69M", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"50-69F_hcc":["m_hcc", "50-69F", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"70+M_hcc":["m_hcc", "70+M", 1990, 2021, 2022,0.313, 0.313, 0.081]},
         {"70+F_hcc":["m_hcc", "70+F", 1990, 2021, 2022,0.313, 0.313, 0.081]}]

## Treatment Effectiveness worst case
tex_wc=[ #HBeAg seroclearance
        {"0-4M_te_icl_ict":["te_icl_ict", "0-4M", 1990, 2021, 2022,2,2,1]},
        {"0-4F_te_icl_ict":["te_icl_ict", "0-4F", 1990, 2021, 2022,2,2,1]},
        {"5-14M_te_icl_ict":["te_icl_ict", "5-14M", 1990, 2021, 2022,2,2,1]},
        {"5-14F_te_icl_ict":["te_icl_ict", "5-14F", 1990, 2021, 2022,2,2,1]},
        {"15-49M_te_icl_ict":["te_icl_ict", "15-49M", 1990, 2021, 2022,2,2,1]},
        {"15-49F_te_icl_ict":["te_icl_ict", "15-49F", 1990, 2021, 2022,2,2,1]},
        {"50-69M_te_icl_ict":["te_icl_ict", "50-69M", 1990, 2021, 2022,2,2,1]},
        {"50-69F_te_icl_ict":["te_icl_ict", "50-69F", 1990, 2021, 2022,2,2,1]},
        {"70+M_te_icl_ict":["te_icl_ict", "70+M", 1990, 2021, 2022,2,2,1]},
        {"70+F_te_icl_ict":["te_icl_ict", "70+F", 1990, 2021, 2022,2,2,1]},
         #Immune clearance to compensated cirrhosis
        {"0-4M_te_icl_cc":["te_icl_cc", "0-4M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"0-4F_te_icl_cc":["te_icl_cc", "0-4F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"5-14M_te_icl_cc":["te_icl_cc", "5-14M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"5-14F_te_icl_cc":["te_icl_cc", "5-14F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"15-49M_te_icl_cc":["te_icl_cc", "15-49M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"15-49F_te_icl_cc":["te_icl_cc", "15-49F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"50-69M_te_icl_cc":["te_icl_cc", "50-69M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"50-69F_te_icl_cc":["te_icl_cc", "50-69F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"70+M_te_icl_cc":["te_icl_cc", "70+M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"70+F_te_icl_cc":["te_icl_cc", "70+F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        #Immune escape to compensated cirrhosis
        {"0-4M_te_ie_cc":["te_ie_cc", "0-4M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"0-4F_te_ie_cc":["te_ie_cc", "0-4F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"5-14M_te_ie_cc":["te_ie_cc", "5-14M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"5-14F_te_ie_cc":["te_ie_cc", "5-14F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"15-49M_te_ie_cc":["te_ie_cc", "15-49M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"15-49F_te_ie_cc":["te_ie_cc", "15-49F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"50-69M_te_ie_cc":["te_ie_cc", "50-69M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"50-69F_te_ie_cc":["te_ie_cc", "50-69F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"70+M_te_ie_cc":["te_ie_cc", "70+M", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        {"70+F_te_ie_cc":["te_ie_cc", "70+F", 1990, 2021, 2022, 0.2, 0.2, 0.4]},
        #Compensated cirrhosis to decompensated cirrhosis
        {"0-4M_te_cc_dc":["te_cc_dc", "0-4M", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"0-4F_te_cc_dc":["te_cc_dc", "0-4F", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"5-14M_te_cc_dc":["te_cc_dc", "5-14M", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"5-14F_te_cc_dc":["te_cc_dc", "5-14F", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"15-49M_te_cc_dc":["te_cc_dc", "15-49M", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"15-49F_te_cc_dc":["te_cc_dc", "15-49F", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"50-69M_te_cc_dc":["te_cc_dc", "50-69M", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"50-69F_te_cc_dc":["te_cc_dc", "50-69F", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"70+M_te_cc_dc":["te_cc_dc", "70+M", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        {"70+F_te_cc_dc":["te_cc_dc", "70+M", 1990, 2021, 2022, 0.5, 0.5, 0.75]},
        #Decompensated cirrhosis to compensated cirrhosis
        {"0-4M_te_dc_cc":["te_dc_cc", "0-4M", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"0-4F_te_dc_cc":["te_dc_cc", "0-4F", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"5-14M_te_dc_cc":["te_dc_cc", "5-14M", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"5-14F_te_dc_cc":["te_dc_cc", "5-14F", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"15-49M_te_dc_cc":["te_dc_cc", "15-49M", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"15-49F_te_dc_cc":["te_dc_cc", "15-49F", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"50-69M_te_dc_cc":["te_dc_cc", "50-69M", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"50-69F_te_dc_cc":["te_dc_cc", "50-69F", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"70+M_te_dc_cc":["te_dc_cc", "70+M", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        {"70+F_te_dc_cc":["te_dc_cc", "70+F", 1990, 2021, 2022,0.15, 0.15, 0.1]},
        #Immune clearance to hepatocellular carcinoma
        {"0-4M_te_icl_hcc":["te_icl_hcc", "0-4M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"0-4F_te_icl_hcc":["te_icl_hcc", "0-4F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"5-14M_te_icl_hcc":["te_icl_hcc", "5-14M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"5-14F_te_icl_hcc":["te_icl_hcc", "5-14F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"15-49M_te_icl_hcc":["te_icl_hcc", "15-49M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"15-49F_te_icl_hcc":["te_icl_hcc", "15-49F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"50-69M_te_icl_hcc":["te_icl_hcc", "50-69M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"50-69F_te_icl_hcc":["te_icl_hcc", "50-69F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"70+M_te_icl_hcc":["te_icl_hcc", "70+M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"70+F_te_icl_hcc":["te_icl_hcc", "70+F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        #Immune control to hepatocellular carcinoma
        {"0-4M_te_ict_hcc":["te_ict_hcc", "0-4M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"0-4F_te_ict_hcc":["te_ict_hcc", "0-4F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"5-14M_te_ict_hcc":["te_ict_hcc", "5-14M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"5-14F_te_ict_hcc":["te_ict_hcc", "5-14F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"15-49M_te_ict_hcc":["te_ict_hcc", "15-49M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"15-49F_te_ict_hcc":["te_ict_hcc", "15-49F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"50-69M_te_ict_hcc":["te_ict_hcc", "50-69M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"50-69F_te_ict_hcc":["te_ict_hcc", "50-69F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"70+M_te_ict_hcc":["te_ict_hcc", "70+M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"70+F_te_ict_hcc":["te_ict_hcc", "70+F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        #Immune escape to hepatocellular carcinoma
        {"0-4M_te_ie_hcc":["te_ie_hcc", "0-4M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"0-4F_te_ie_hcc":["te_ie_hcc", "0-4F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"5-14M_te_ie_hcc":["te_ie_hcc", "5-14M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"5-14F_te_ie_hcc":["te_ie_hcc", "5-14F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"15-49M_te_ie_hcc":["te_ie_hcc", "15-49M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"15-49F_te_ie_hcc":["te_ie_hcc", "15-49F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"50-69M_te_ie_hcc":["te_ie_hcc", "50-69M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"50-69F_te_ie_hcc":["te_ie_hcc", "50-69F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"70+M_te_ie_hcc":["te_ie_hcc", "70+M", 1990, 2021, 2022, 0.2,0.2,0.5]},
        {"70+F_te_ie_hcc":["te_ie_hcc", "70+F", 1990, 2021, 2022, 0.2,0.2,0.5]},
        #Compensated cirrhosis to hepatocellular carcinoma
        {"0-4M_te_cc_hcc":["te_cc_hcc", "0-4M", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"0-4F_te_cc_hcc":["te_cc_hcc", "0-4F", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"5-14M_te_cc_hcc":["te_cc_hcc", "5-14M", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"5-14F_te_cc_hcc":["te_cc_hcc", "5-14F", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"15-49M_te_cc_hcc":["te_cc_hcc", "15-49M", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"15-49F_te_cc_hcc":["te_cc_hcc", "15-49F", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"50-69M_te_cc_hcc":["te_cc_hcc", "50-69M", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"50-69F_te_cc_hcc":["te_cc_hcc", "50-69F", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"70+M_te_cc_hcc":["te_cc_hcc", "70+M", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        {"70+F_te_cc_hcc":["te_cc_hcc", "70+F", 1990, 2021, 2022, 0.7, 0.7, 0.85]},
        #Decompensated cirrhosis to hepatocellular carcinoma
        {"0-4M_te_dc_hcc":["te_dc_hcc", "0-4M", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"0-4F_te_dc_hcc":["te_dc_hcc", "0-4F", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"5-14M_te_dc_hcc":["te_dc_hcc", "5-14M", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"5-14F_te_dc_hcc":["te_dc_hcc", "5-14F", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"15-49M_te_dc_hcc":["te_dc_hcc", "15-49M", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"15-49F_te_dc_hcc":["te_dc_hcc", "15-49F", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"50-69M_te_dc_hcc":["te_dc_hcc", "50-69M", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"50-69F_te_dc_hcc":["te_dc_hcc", "50-69F", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"70+M_te_dc_hcc":["te_dc_hcc", "70+M", 1990, 2021, 2022, 0.75, 0.75, 1]},
        {"70+F_te_dc_hcc":["te_dc_hcc", "70+F", 1990, 2021, 2022, 0.75, 0.75, 1]},
        #Mortality from decompensated cirrhosis
        {"0-4M_te_m_dc":["te_m_dc", "0-4M", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"0-4F_te_m_dc":["te_m_dc", "0-4F", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"5-14M_te_m_dc":["te_m_dc", "5-14M", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"5-14F_te_m_dc":["te_m_dc", "5-14F", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"15-49M_te_m_dc":["te_m_dc", "15-49M", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"15-49F_te_m_dc":["te_m_dc", "15-49F", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"50-69M_te_m_dc":["te_m_dc", "50-69M", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"50-69F_te_m_dc":["te_m_dc", "50-69F", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"70+M_te_m_dc":["te_m_dc", "70+M", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        {"70+F_te_m_dc":["te_m_dc", "70+F", 1990, 2021, 2022, 0.5, 0.5, 0.7]},
        #Mortality from hepatocellular carcinoma
        {"0-4M_te_m_hcc":["te_m_hcc", "0-4M", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"0-4F_te_m_hcc":["te_m_hcc", "0-4F", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"5-14M_te_m_hcc":["te_m_hcc", "5-14M", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"5-14F_te_m_hcc":["te_m_hcc", "5-14F", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"15-49M_te_m_hcc":["te_m_hcc", "15-49M", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"15-49F_te_m_hcc":["te_m_hcc", "15-49F", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"50-69M_te_m_hcc":["te_m_hcc", "50-69M", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"50-69F_te_m_hcc":["te_m_hcc", "50-69F", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"70+M_te_m_hcc":["te_m_hcc", "70+M", 1990, 2021, 2022, 0.8, 0.8, 1]},
        {"70+F_te_m_hcc":["te_m_hcc", "70+F", 1990, 2021, 2022, 0.8, 0.8, 1]}
        ]

## Treatment Effectiveness best case
tex_bc=[ #HBeAg seroclearance
        {"0-4M_te_icl_ict":["te_icl_ict", "0-4M", 1990, 2021, 2022,2,2,3]},
        {"0-4F_te_icl_ict":["te_icl_ict", "0-4F", 1990, 2021, 2022,2,2,3]},
        {"5-14M_te_icl_ict":["te_icl_ict", "5-14M", 1990, 2021, 2022,2,2,3]},
        {"5-14F_te_icl_ict":["te_icl_ict", "5-14F", 1990, 2021, 2022,2,2,3]},
        {"15-49M_te_icl_ict":["te_icl_ict", "15-49M", 1990, 2021, 2022,2,2,3]},
        {"15-49F_te_icl_ict":["te_icl_ict", "15-49F", 1990, 2021, 2022,2,2,3]},
        {"50-69M_te_icl_ict":["te_icl_ict", "50-69M", 1990, 2021, 2022,2,2,3]},
        {"50-69F_te_icl_ict":["te_icl_ict", "50-69F", 1990, 2021, 2022,2,2,3]},
        {"70+M_te_icl_ict":["te_icl_ict", "70+M", 1990, 2021, 2022,2,2,3]},
        {"70+F_te_icl_ict":["te_icl_ict", "70+F", 1990, 2021, 2022,2,2,3]},
         #Immune clearance to compensated cirrhosis
        {"0-4M_te_icl_cc":["te_icl_cc", "0-4M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"0-4F_te_icl_cc":["te_icl_cc", "0-4F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"5-14M_te_icl_cc":["te_icl_cc", "5-14M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"5-14F_te_icl_cc":["te_icl_cc", "5-14F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"15-49M_te_icl_cc":["te_icl_cc", "15-49M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"15-49F_te_icl_cc":["te_icl_cc", "15-49F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"50-69M_te_icl_cc":["te_icl_cc", "50-69M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"50-69F_te_icl_cc":["te_icl_cc", "50-69F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"70+M_te_icl_cc":["te_icl_cc", "70+M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"70+F_te_icl_cc":["te_icl_cc", "70+F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        #Immune escape to compensated cirrhosis
        {"0-4M_te_ie_cc":["te_ie_cc", "0-4M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"0-4F_te_ie_cc":["te_ie_cc", "0-4F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"5-14M_te_ie_cc":["te_ie_cc", "5-14M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"5-14F_te_ie_cc":["te_ie_cc", "5-14F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"15-49M_te_ie_cc":["te_ie_cc", "15-49M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"15-49F_te_ie_cc":["te_ie_cc", "15-49F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"50-69M_te_ie_cc":["te_ie_cc", "50-69M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"50-69F_te_ie_cc":["te_ie_cc", "50-69F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"70+M_te_ie_cc":["te_ie_cc", "70+M", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        {"70+F_te_ie_cc":["te_ie_cc", "70+F", 1990, 2021, 2022, 0.2, 0.2, 0.1]},
        #Compensated cirrhosis to decompensated cirrhosis
        {"0-4M_te_cc_dc":["te_cc_dc", "0-4M", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"0-4F_te_cc_dc":["te_cc_dc", "0-4F", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"5-14M_te_cc_dc":["te_cc_dc", "5-14M", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"5-14F_te_cc_dc":["te_cc_dc", "5-14F", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"15-49M_te_cc_dc":["te_cc_dc", "15-49M", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"15-49F_te_cc_dc":["te_cc_dc", "15-49F", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"50-69M_te_cc_dc":["te_cc_dc", "50-69M", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"50-69F_te_cc_dc":["te_cc_dc", "50-69F", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"70+M_te_cc_dc":["te_cc_dc", "70+M", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        {"70+F_te_cc_dc":["te_cc_dc", "70+M", 1990, 2021, 2022, 0.5, 0.5, 0.35]},
        #Decompensated cirrhosis to compensated cirrhosis
        {"0-4M_te_dc_cc":["te_dc_cc", "0-4M", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"0-4F_te_dc_cc":["te_dc_cc", "0-4F", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"5-14M_te_dc_cc":["te_dc_cc", "5-14M", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"5-14F_te_dc_cc":["te_dc_cc", "5-14F", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"15-49M_te_dc_cc":["te_dc_cc", "15-49M", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"15-49F_te_dc_cc":["te_dc_cc", "15-49F", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"50-69M_te_dc_cc":["te_dc_cc", "50-69M", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"50-69F_te_dc_cc":["te_dc_cc", "50-69F", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"70+M_te_dc_cc":["te_dc_cc", "70+M", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        {"70+F_te_dc_cc":["te_dc_cc", "70+F", 1990, 2021, 2022,0.15, 0.15, 0.25]},
        #Immune clearance to hepatocellular carcinoma
        {"0-4M_te_icl_hcc":["te_icl_hcc", "0-4M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"0-4F_te_icl_hcc":["te_icl_hcc", "0-4F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"5-14M_te_icl_hcc":["te_icl_hcc", "5-14M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"5-14F_te_icl_hcc":["te_icl_hcc", "5-14F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"15-49M_te_icl_hcc":["te_icl_hcc", "15-49M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"15-49F_te_icl_hcc":["te_icl_hcc", "15-49F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"50-69M_te_icl_hcc":["te_icl_hcc", "50-69M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"50-69F_te_icl_hcc":["te_icl_hcc", "50-69F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"70+M_te_icl_hcc":["te_icl_hcc", "70+M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"70+F_te_icl_hcc":["te_icl_hcc", "70+F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        #Immune control to hepatocellular carcinoma
        {"0-4M_te_ict_hcc":["te_ict_hcc", "0-4M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"0-4F_te_ict_hcc":["te_ict_hcc", "0-4F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"5-14M_te_ict_hcc":["te_ict_hcc", "5-14M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"5-14F_te_ict_hcc":["te_ict_hcc", "5-14F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"15-49M_te_ict_hcc":["te_ict_hcc", "15-49M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"15-49F_te_ict_hcc":["te_ict_hcc", "15-49F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"50-69M_te_ict_hcc":["te_ict_hcc", "50-69M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"50-69F_te_ict_hcc":["te_ict_hcc", "50-69F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"70+M_te_ict_hcc":["te_ict_hcc", "70+M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"70+F_te_ict_hcc":["te_ict_hcc", "70+F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        #Immune escape to hepatocellular carcinoma
        {"0-4M_te_ie_hcc":["te_ie_hcc", "0-4M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"0-4F_te_ie_hcc":["te_ie_hcc", "0-4F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"5-14M_te_ie_hcc":["te_ie_hcc", "5-14M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"5-14F_te_ie_hcc":["te_ie_hcc", "5-14F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"15-49M_te_ie_hcc":["te_ie_hcc", "15-49M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"15-49F_te_ie_hcc":["te_ie_hcc", "15-49F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"50-69M_te_ie_hcc":["te_ie_hcc", "50-69M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"50-69F_te_ie_hcc":["te_ie_hcc", "50-69F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"70+M_te_ie_hcc":["te_ie_hcc", "70+M", 1990, 2021, 2022, 0.2,0.2,0.05]},
        {"70+F_te_ie_hcc":["te_ie_hcc", "70+F", 1990, 2021, 2022, 0.2,0.2,0.05]},
        #Compensated cirrhosis to hepatocellular carcinoma
        {"0-4M_te_cc_hcc":["te_cc_hcc", "0-4M", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"0-4F_te_cc_hcc":["te_cc_hcc", "0-4F", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"5-14M_te_cc_hcc":["te_cc_hcc", "5-14M", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"5-14F_te_cc_hcc":["te_cc_hcc", "5-14F", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"15-49M_te_cc_hcc":["te_cc_hcc", "15-49M", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"15-49F_te_cc_hcc":["te_cc_hcc", "15-49F", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"50-69M_te_cc_hcc":["te_cc_hcc", "50-69M", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"50-69F_te_cc_hcc":["te_cc_hcc", "50-69F", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"70+M_te_cc_hcc":["te_cc_hcc", "70+M", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        {"70+F_te_cc_hcc":["te_cc_hcc", "70+F", 1990, 2021, 2022, 0.7, 0.7, 0.5]},
        #Decompensated cirrhosis to hepatocellular carcinoma
        {"0-4M_te_dc_hcc":["te_dc_hcc", "0-4M", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"0-4F_te_dc_hcc":["te_dc_hcc", "0-4F", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"5-14M_te_dc_hcc":["te_dc_hcc", "5-14M", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"5-14F_te_dc_hcc":["te_dc_hcc", "5-14F", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"15-49M_te_dc_hcc":["te_dc_hcc", "15-49M", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"15-49F_te_dc_hcc":["te_dc_hcc", "15-49F", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"50-69M_te_dc_hcc":["te_dc_hcc", "50-69M", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"50-69F_te_dc_hcc":["te_dc_hcc", "50-69F", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"70+M_te_dc_hcc":["te_dc_hcc", "70+M", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        {"70+F_te_dc_hcc":["te_dc_hcc", "70+F", 1990, 2021, 2022, 0.75, 0.75, 0.5]},
        #Mortality from decompensated cirrhosis
        {"0-4M_te_m_dc":["te_m_dc", "0-4M", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"0-4F_te_m_dc":["te_m_dc", "0-4F", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"5-14M_te_m_dc":["te_m_dc", "5-14M", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"5-14F_te_m_dc":["te_m_dc", "5-14F", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"15-49M_te_m_dc":["te_m_dc", "15-49M", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"15-49F_te_m_dc":["te_m_dc", "15-49F", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"50-69M_te_m_dc":["te_m_dc", "50-69M", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"50-69F_te_m_dc":["te_m_dc", "50-69F", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"70+M_te_m_dc":["te_m_dc", "70+M", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        {"70+F_te_m_dc":["te_m_dc", "70+F", 1990, 2021, 2022, 0.5, 0.5, 0.25]},
        #Mortality from hepatocellular carcinoma
        {"0-4M_te_m_hcc":["te_m_hcc", "0-4M", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"0-4F_te_m_hcc":["te_m_hcc", "0-4F", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"5-14M_te_m_hcc":["te_m_hcc", "5-14M", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"5-14F_te_m_hcc":["te_m_hcc", "5-14F", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"15-49M_te_m_hcc":["te_m_hcc", "15-49M", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"15-49F_te_m_hcc":["te_m_hcc", "15-49F", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"50-69M_te_m_hcc":["te_m_hcc", "50-69M", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"50-69F_te_m_hcc":["te_m_hcc", "50-69F", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"70+M_te_m_hcc":["te_m_hcc", "70+M", 1990, 2021, 2022, 0.8, 0.8, 0.5]},
        {"70+F_te_m_hcc":["te_m_hcc", "70+F", 1990, 2021, 2022, 0.8, 0.8, 0.5]}
        ]

def par_sens(F, db_bl, db_s1, calib, par_dict):
    
    out_res=[{"pop":["alive", "total", "sum"]}, #total population size
             {"pop_u5":["alive", {"Under 5":["0-4M", "0-4F"]}, "sum"]}, #total under 5 population size
             {"prev":["prev", "total", "weighted"]}, #total population prevalence
             {"chb_pop":["chb_pop", "total", "weighted"]}, #total population living with CHB
             {"prev_u5":["prev", {"Under 5":["0-4M", "0-4F"]}, "weighted"]}, #under 5y prevalence
             {"mort":[":dd_hbv", "total", "sum"]}, #Total HBV mortality
             {"hcc_inc": ["flw_hcc", "total", "sum"]}, #HCC incidence
             {"chb_inc": ["tot_inc", "total", "sum"]}, #CHB incidence
             {"hbe_preg": ["eag_ott", "15-49F", "sum"]}, #HBeAg prevalence in pregnant women
             {"yld": ["yld", "total", "sum"]}, #years lost to disability
             {"births": ["b_rate",{"Under 5":["0-4M", "0-4F"]}, "sum" ]}, #total births
             {"bd_cov": ["bd", {"Under 5":["0-4M", "0-4F"]}, "weighted"]}, #birth dose coverage
             {"hb3_cov": ["hb3", {"Under 5":["0-4M", "0-4F"]}, "weighted"]}, #hb3 coverage
             {"dx_rate":["tot_dx", "total", "sum"]}, #annual (incident) diagnoses
             {"tx_cov": ["treat", "total", "sum"]}, #total treatment coverage
             {"dm_dx": [[{"dm_dx":"it_dx+icl_dx+ict_dx+ie_dx+cc_dx+dc_dx+hcc_dx"}], "total", "sum"]}, #disease management, no treatment (for costs)
             {"dm_tx":[[{"dm_tx":"icl_tx+ict_tx+ie_tx+cc_tx+dc_tx+hcc_tx"}], "total", "sum"]}, #disease management, treatment (for costs)
             {"pop_hcc": [[{"pop_hcc":"cc_dx+cc_tx+dc_dx+dc_tx"}], "total", "sum"]}, #population HCC surveillance (for costs)
             {"tgt_hcc": [[{"tgt_hcc":"it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"50+":["50-69M", "50-69F", "70+M", "70+F"]}, "sum"]}, #targeted HCC surveillance (for costs)
             {"tgt_hcc_b": [[{"tgt_hcc":"it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"40+":["15-49M", "15-49F"]}, "sum"]}, #targeted HCC surveillance (for costs (40-49))
             {"hsp_tx": [[{"hsp_tx":"dc_tx+hcc_tx"}], "total", "sum"]}, #hospitalizations among treated (for costs)
             {"hsp_utx":[[{"hsp_utx":"dc+dc_dx+hcc+hcc_dx"}], "total", "sum"]},#hospitalization among untreated (for costs)
             {"dx_prop": ["diag_cov", "total", "weighted"]}, #diagnosis coverage
             {"tx_prop": ["treat_cov", "total", "weighted"]}, #treatment coverage (among eligible) 
             {"mav_n": ["mav_births",{"Under 5":["0-4M", "0-4F"]}, "sum" ]}, #number of births recieving mAVs and HBIG,
             {"prg_scr": ["preg_scr_num", "15-49F", "sum"]}, #number of pregnant women screened for HBsAg,
             {"prg_hrs":["preg_scr_num", "15-49F", "sum"]}, #number of pregnant women screened for HBeAg
             {"fc_cov_val": ["fc_cov", "total", "weighted"]}] #FC coverage among treated population
    
    D_bl=at.ProjectData.from_spreadsheet("databooks/"+db_bl, framework=F)
    P_bl=at.Project(framework=F, databook="databooks/"+db_bl, sim_start=1990, sim_end=2100, sim_dt=0.25, do_run=False)
    cal_bl=P_bl.make_parset()
    cal_bl.load_calibration("calibrations/"+calib)

    bl=P_bl.run_sim(parset=cal_bl, result_name = "Status Quo")
    
    par_scen_bl=at.ParameterScenario(name="scen_bl")
    
    for i in par_dict:
        for key,val in i.items():
            par_scen_bl.add(val[0], val[1], [val[2], val[3], val[4]], [val[5], val[6], val[7]])
    
    par_scen_bl_res=par_scen_bl.run(P_bl, parset=cal_bl)
    
    central_est_bl={}
    for i in out_res:
        for key, val in i.items():
            df=at.PlotData(par_scen_bl_res, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1).series[0].vals
            central_est_bl[key]=df
    
    
    D_s1=at.ProjectData.from_spreadsheet("databooks/"+db_s1, framework=F)
    P_s1=at.Project(framework=F, databook="databooks/"+db_s1, sim_start=1990, sim_end=2100, sim_dt=0.25, do_run=False)
    cal_s1=P_s1.make_parset()
    cal_s1.load_calibration("calibrations/"+calib)

    s1=P_s1.run_sim(parset=cal_s1, result_name = "Scenario 1: 2030 Target")
    
    par_scen_s1=at.ParameterScenario(name="scen_s1")
    
    for i in par_dict:
        for key,val in i.items():
            par_scen_s1.add(val[0], val[1], [val[2], val[3], val[4]], [val[5], val[6], val[7]])
    
    par_scen_s1_res=par_scen_s1.run(P_s1, parset=cal_s1)
    
    central_est_s1={}
    for i in out_res:
        for key, val in i.items():
            df=at.PlotData(par_scen_s1_res, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1).series[0].vals
            central_est_s1[key]=df
    
    nan_keys = ["bd_cov", "hb3_cov", "dx_prop", "tx_prop", "fc_cov_val", "prev", "prev_u5"]
    for k in nan_keys:
        if k in central_est_bl:
            central_est_bl[k] = np.nan_to_num(central_est_bl[k], nan=0.0)
        if k in central_est_s1:
            central_est_s1[k] = np.nan_to_num(central_est_s1[k], nan=0.0)
    
    return central_est_bl, central_est_s1
    
## MTCT worst case
afr_mtctw_bl, afr_mtctw_s1= par_sens(F, "AFR.xlsx", "AFR_S1.xlsx", "AFR_calib.xlsx", mtct_wc)
amr_mtctw_bl, amr_mtctw_s1= par_sens(F, "AMR.xlsx", "AMR_S1.xlsx", "AMR_calib.xlsx", mtct_wc)
emr_mtctw_bl, emr_mtctw_s1= par_sens(F, "EMR.xlsx", "EMR_S1.xlsx", "EMR_calib.xlsx", mtct_wc)
eur_mtctw_bl, eur_mtctw_s1= par_sens(F, "EUR.xlsx", "EUR_S1.xlsx", "EUR_calib.xlsx", mtct_wc)
sear_mtctw_bl, sear_mtctw_s1= par_sens(F, "SEAR.xlsx", "SEAR_S1.xlsx", "SEAR_calib.xlsx", mtct_wc)
wpr_mtctw_bl, wpr_mtctw_s1= par_sens(F, "WPR.xlsx", "WPR_S1.xlsx", "WPR_calib.xlsx", mtct_wc)

## MTCT best case
afr_mtctb_bl, afr_mtctb_s1=  par_sens(F, "AFR.xlsx", "AFR_S1.xlsx", "AFR_calib.xlsx", mtct_bc)
amr_mtctb_bl, amr_mtctb_s1=par_sens(F, "AMR.xlsx", "AMR_S1.xlsx", "AMR_calib.xlsx", mtct_bc)
emr_mtctb_bl, emr_mtctb_s1= par_sens(F, "EMR.xlsx", "EMR_S1.xlsx", "EMR_calib.xlsx", mtct_bc)
eur_mtctb_bl, eur_mtctb_s1= par_sens(F, "EUR.xlsx", "EUR_S1.xlsx", "EUR_calib.xlsx", mtct_bc)
sear_mtctb_bl, sear_mtctb_s1= par_sens(F, "SEAR.xlsx", "SEAR_S1.xlsx", "SEAR_calib.xlsx", mtct_bc)
wpr_mtctb_bl, wpr_mtctb_s1= par_sens(F, "WPR.xlsx", "WPR_S1.xlsx", "WPR_calib.xlsx", mtct_bc)

## Mortality worst case
afr_mortw_bl, afr_mortw_s1= par_sens(F, "AFR.xlsx", "AFR_S1.xlsx", "AFR_calib.xlsx", mort_wc)
amr_mortw_bl, amr_mortw_s1= par_sens(F, "AMR.xlsx", "AMR_S1.xlsx", "AMR_calib.xlsx", mort_wc)
emr_mortw_bl, emr_mortw_s1= par_sens(F, "EMR.xlsx", "EMR_S1.xlsx", "EMR_calib.xlsx", mort_wc)
eur_mortw_bl, eur_mortw_s1=  par_sens(F, "EUR.xlsx", "EUR_S1.xlsx", "EUR_calib.xlsx", mort_wc)
sear_mortw_bl, sear_mortw_s1= par_sens(F, "SEAR.xlsx", "SEAR_S1.xlsx", "SEAR_calib.xlsx", mort_wc)
wpr_mortw_bl, wpr_mortw_s1=par_sens(F, "WPR.xlsx", "WPR_S1.xlsx", "WPR_calib.xlsx", mort_wc)

## Mortality best case
afr_mortb_bl, afr_mortb_s1=  par_sens(F, "AFR.xlsx", "AFR_S1.xlsx", "AFR_calib.xlsx", mort_bc)
amr_mortb_bl, amr_mortb_s1= par_sens(F, "AMR.xlsx", "AMR_S1.xlsx", "AMR_calib.xlsx", mort_bc)
emr_mortb_bl, emr_mortb_s1= par_sens(F, "EMR.xlsx", "EMR_S1.xlsx", "EMR_calib.xlsx", mort_bc)
eur_mortb_bl, eur_mortb_s1=  par_sens(F, "EUR.xlsx", "EUR_S1.xlsx", "EUR_calib.xlsx", mort_bc)
sear_mortb_bl, sear_mortb_s1=  par_sens(F, "SEAR.xlsx", "SEAR_S1.xlsx", "SEAR_calib.xlsx", mort_bc)
wpr_mortb_bl, wpr_mortb_s1=par_sens(F, "WPR.xlsx", "WPR_S1.xlsx", "WPR_calib.xlsx", mort_bc)

## Treatment worst case (dis prog)
afr_trtw_bl, afr_trtw_s1= par_sens(F, "AFR.xlsx", "AFR_S1.xlsx", "AFR_calib.xlsx", tex_wc)
amr_trtw_bl, amr_trtw_s1= par_sens(F, "AMR.xlsx", "AMR_S1.xlsx", "AMR_calib.xlsx", tex_wc)
emr_trtw_bl, emr_trtw_s1= par_sens(F, "EMR.xlsx", "EMR_S1.xlsx", "EMR_calib.xlsx", tex_wc)
eur_trtw_bl, eur_trtw_s1=  par_sens(F, "EUR.xlsx", "EUR_S1.xlsx", "EUR_calib.xlsx", tex_wc)
sear_trtw_bl, sear_trtw_s1=  par_sens(F, "SEAR.xlsx", "SEAR_S1.xlsx", "SEAR_calib.xlsx", tex_wc)
wpr_trtw_bl, wpr_trtw_s1=par_sens(F, "WPR.xlsx", "WPR_S1.xlsx", "WPR_calib.xlsx", tex_wc)


## Treatment best case (dis prog)
afr_trtb_bl, afr_trtb_s1=  par_sens(F, "AFR.xlsx", "AFR_S1.xlsx", "AFR_calib.xlsx", tex_bc)
amr_trtb_bl, amr_trtb_s1= par_sens(F, "AMR.xlsx", "AMR_S1.xlsx", "AMR_calib.xlsx", tex_bc)
emr_trtb_bl, emr_trtb_s1= par_sens(F, "EMR.xlsx", "EMR_S1.xlsx", "EMR_calib.xlsx", tex_bc)
eur_trtb_bl, eur_trtb_s1=  par_sens(F, "EUR.xlsx", "EUR_S1.xlsx", "EUR_calib.xlsx", tex_bc)
sear_trtb_bl, sear_trtb_s1=  par_sens(F, "SEAR.xlsx", "SEAR_S1.xlsx", "SEAR_calib.xlsx", tex_bc)
wpr_trtb_bl, wpr_trtb_s1=par_sens(F, "WPR.xlsx", "WPR_S1.xlsx", "WPR_calib.xlsx", tex_bc)

def parsens_econ(psens_bl, psens_s1, reg, wd, cost_data, disc_rate, fc_cost=5000):
    
    # psens_bl=amr_mtctw_bl
    # psens_s1=amr_mtctw_s1
    # reg="AMR"
    # disc_rate=0.015
    # cost_data="cost and agg data/costs.xlsx"
    
    ## FIX: defensively sanitize Atomica-derived params (NaN propagates through cost/DALY calcs)
    _sanitize_keys = ("mort", "yld", "dm_tx", "dm_dx", "fc_cov_val", "tx_cov", "dx_rate",
                      "dx_prop", "pop", "births", "bd_cov", "hb3_cov", "mav_n", "prg_scr",
                      "prg_hrs", "hsp_utx", "hsp_tx", "pop_hcc", "tgt_hcc", "tgt_hcc_b",
                      "chb_inc", "hcc_inc")
    for d in (psens_bl, psens_s1):
        for k in _sanitize_keys:
            if k in d:
                d[k] = np.nan_to_num(d[k], nan=0.0, posinf=0.0, neginf=0.0)
    
    #Discounting Array
    if disc_rate >1:
        disc_rate=disc_rate/100
    else:
        disc_rate=disc_rate
        
    discount=np.zeros((len(np.arange(1990,2100,1)),3))
    discount[:,0]=np.arange(1990,2100,1)
    
    for idx,val in enumerate(discount[:,0]):
        if val <= 2025:
            discount[idx,1]=1
            discount[idx,2]=0 #use this for high risk screening as a check
        else:
            discount[idx,1:]=(1+disc_rate)**(2025-val)
    
    
    vax_costs=pd.read_excel(cost_data, sheet_name="vax")
    bd_vax=vax_costs[reg].iloc[0]
    hb3_vax=vax_costs[reg].iloc[1]
    hrs_cost=vax_costs[reg].iloc[10]
    mav_cost=vax_costs[reg].iloc[11]
    
    care_costs=pd.read_excel(cost_data, sheet_name="care")
    dx_cost=care_costs[reg].iloc[0]
    tx_cost=care_costs[reg].iloc[3]
    dx_dmc=care_costs[reg].iloc[4] #cost of disease management (diagnosed)
    tx_dmc=care_costs[reg].iloc[4] #cost of disease management (on treatment)
    hosp_cost=care_costs[reg].iloc[7] #cost of hospitalisation
    hcc_cost=care_costs[reg].iloc[6] #HCC surveillance costs
    hcc_prp=care_costs[reg].iloc[8] #proportion of CHB in 15-49y who are 40-49y (HCC screening)
    
    ## Vaccines and mAV
    
    bd_cost_bl, hb3_cost_bl, bd_cost_s1, hb3_cost_s1=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    bd_cost_bl[:,0]=psens_bl["bd_cov"][:]*psens_bl["births"][:]*bd_vax*discount[:,1]
    hb3_cost_bl[:,0]=psens_bl["hb3_cov"][:]*psens_bl["births"][:]*hb3_vax*discount[:,1]
    bd_cost_s1[:,0]=psens_s1["bd_cov"][:]*psens_s1["births"][:]*bd_vax*discount[:,1]
    hb3_cost_s1[:,0]=psens_s1["hb3_cov"][:]*psens_s1["births"][:]*hb3_vax*discount[:,1]
    
    bl_cost_mav, s1_cost_mav=np.zeros((110,1)),np.zeros((110,1))
    bl_cost_mav[:,0]=(psens_bl["prg_scr"][:]*dx_cost*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    s1_cost_mav[:,0]=(psens_s1["mav_n"][:]*mav_cost*discount[:,1])+(psens_s1["prg_scr"][:]*dx_cost*discount[:,1])+(psens_s1["prg_hrs"][:]*hrs_cost*discount[:,2])
    
    ## Diagnoses
    bl_dx_inc, s1_dx_inc=np.zeros((110,1)),np.zeros((110,1))
    
    for i in range(len(psens_bl["dx_prop"])):
        if i < 1:
            bl_dx_inc[i,0]=psens_bl["dx_prop"][i]
            s1_dx_inc[i,0]=psens_s1["dx_prop"][i]
        else:
            bl_dx_inc[i,0]=max(psens_bl["dx_prop"][i]-psens_bl["dx_prop"][i-1],0)
            s1_dx_inc[i,0]=max(psens_s1["dx_prop"][i]-psens_s1["dx_prop"][i-1],0)
    
    dx_costb_bl, dx_costb_s1=np.zeros((110,1)),np.zeros((110,1))
    
    for yr in range(len(bl_dx_inc)):
        dx_costb_bl[yr,0]=(psens_bl["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*bl_dx_inc[yr,0]*(psens_bl["pop"][yr]-psens_bl["dx_rate"][yr])*discount[yr,1])
        dx_costb_s1[yr,0]=(psens_s1["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*s1_dx_inc[yr,0]*(psens_s1["pop"][yr]-psens_s1["dx_rate"][yr])*discount[yr,1])
   
    ## Treatment
    bl_cost_tx, s1_cost_tx=np.zeros((110,1)),np.zeros((110,1))
    
    bl_cost_tx[:,0]=psens_bl["tx_cov"][:]*tx_cost*discount[:,1]
    s1_cost_tx[:,0]=psens_s1["tx_cov"][:]*tx_cost*discount[:,1]
    
    ## FIX: Functional Cure Drug Costs — use annual coverage INCREMENT (one-time cure logic),
    ## consistent with econ_analysis. Using the coverage level instead of the increment
    ## would re-charge already-cured patients every year, massively over-stating costs
    ## and inflating the ICER in all sensitivity scenarios.
    fc_cost_bl, fc_cost_s1 = np.zeros((110,1)), np.zeros((110,1))
    if "fc_cov_val" in psens_bl and "dm_tx" in psens_bl:
        fc_cov_bl = np.nan_to_num(psens_bl["fc_cov_val"], nan=0.0)
        fc_diff_bl = np.clip(np.diff(fc_cov_bl, prepend=0), 0, None)  # annual new recipients only
        fc_cost_bl[:,0] = psens_bl["dm_tx"][:] * fc_diff_bl * fc_cost * discount[:,1]
    if "fc_cov_val" in psens_s1 and "dm_tx" in psens_s1:
        fc_cov_s1 = np.nan_to_num(psens_s1["fc_cov_val"], nan=0.0)
        fc_diff_s1 = np.clip(np.diff(fc_cov_s1, prepend=0), 0, None)  # annual new recipients only
        fc_cost_s1[:,0] = psens_s1["dm_tx"][:] * fc_diff_s1 * fc_cost * discount[:,1]
    
    dmc_bl=bd_cost_bl+hb3_cost_bl+bl_cost_mav+dx_costb_bl+bl_cost_tx+fc_cost_bl
    dmc_s1=bd_cost_s1+hb3_cost_s1+s1_cost_mav+dx_costb_s1+s1_cost_tx+fc_cost_s1
    
    ## Disease Management
    util=0.25
    tx_hosp=0.5
    
    manc_bl, manc_s1=np.zeros((110,1)),np.zeros((110,1))
    hospc_bl, hospc_s1=np.zeros((110,1)),np.zeros((110,1))
    hccs_bl, hccs_s1=np.zeros((110,1)),np.zeros((110,1))
    
    manc_bl[:,0]=((psens_bl["dm_dx"][:]*dx_dmc*util)+(psens_bl["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    manc_s1[:,0]=((psens_s1["dm_dx"][:]*dx_dmc*util)+(psens_s1["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    
    hospc_bl[:, 0]=((psens_bl["hsp_utx"][:]*hosp_cost*util)+(psens_bl["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    hospc_s1[:, 0]=((psens_s1["hsp_utx"][:]*hosp_cost*util)+(psens_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    
    hccs_bl[:, 0]=((psens_bl["pop_hcc"][:]*hcc_cost*util)+(psens_bl["tgt_hcc"][:]*hcc_cost*util)+(psens_bl["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    hccs_s1[:, 0]=((psens_s1["pop_hcc"][:]*hcc_cost*util)+(psens_s1["tgt_hcc"][:]*hcc_cost*util)+(psens_s1["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    
    imc_bl=manc_bl+hospc_bl+hccs_bl
    imc_s1=manc_s1+hospc_s1+hccs_s1
    
    ## Productivity, YLL, YPLL, DALYs
    prod_costs=pd.read_excel(cost_data, sheet_name="emp_gdp_lex")
    etp_ratio=prod_costs[reg].iloc[0]
    gdp=prod_costs[reg].iloc[1]
    life_exp=prod_costs[reg].iloc[2]
    
    ## GDP growth
    gdp_grw=np.zeros((len(np.arange(1990,2100,1)),4))
    gdp_grw[:,0]=np.arange(1990,2100,1)
    gdp_grw[:,1:4]=gdp
    
    for i,val in enumerate(gdp_grw[:,0]):
        if val>2026:
            gdp_grw[i,1]=gdp_grw[i-1,1]*1.00
            gdp_grw[i,2]=gdp_grw[i-1,2]*1.015
            gdp_grw[i,3]=gdp_grw[i-1,3]*1.03
            
    age_of_deaths=np.array([0.01, 0.031, 0.253, 0.341, 0.365])
    prop_leaving_age_categories=np.array([1/15, 1/15, 1/20, 1/15])
    all_cause_mort=np.array([0.003, 0.0013, 0.0022, 0.0103, (1/life_exp)])
    
    #Baseline
    cbl_deaths=np.zeros((len(psens_bl["mort"]),2))
    cbl_deaths[:,0]=np.arange(1990,2100,1)
    cbl_deaths[:,1]=psens_bl["mort"]
    
    for idx,val in enumerate(cbl_deaths[:,0]):
        if val < 2026:
            cbl_deaths[idx,1]=0
    
    ghosts_cbl=np.zeros((len(cbl_deaths), len(age_of_deaths)))
    ghosts_cbl[0,:]=cbl_deaths[0,1]*age_of_deaths
    
    for t in range(1,len(cbl_deaths)):
        ppl_who_age=ghosts_cbl[t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
        ghosts_cbl[t,0]=max(0, ghosts_cbl[t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_cbl[t-1,0])
        ghosts_cbl[t,1]=max(0, ghosts_cbl[t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_cbl[t-1,1])
        ghosts_cbl[t,2]=max(0, ghosts_cbl[t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_cbl[t-1,2])
        ghosts_cbl[t,3]=max(0, ghosts_cbl[t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_cbl[t-1,3])
        ghosts_cbl[t,4]=max(0, ghosts_cbl[t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_cbl[t-1,4])
        
        ghosts_cbl[t,:]= ghosts_cbl[t,:]+cbl_deaths[t,1]*age_of_deaths
    
    ## Scenario 1 (central)
    cs1_deaths=np.zeros((len(psens_s1["mort"]),2))
    cs1_deaths[:,0]=np.arange(1990,2100,1)
    cs1_deaths[:,1]=psens_s1["mort"]
     
    for idx,val in enumerate(cs1_deaths[:,0]):
        if val < 2026:
            cs1_deaths[idx,1]=0
     
    ghosts_cs1=np.zeros((len(cs1_deaths), len(age_of_deaths)))
    ghosts_cs1[0,:]=cs1_deaths[0,1]*age_of_deaths
     
    for t in range(1,len(cs1_deaths)):
        ppl_who_age=ghosts_cs1[t,0:len(prop_leaving_age_categories)]*prop_leaving_age_categories
        ghosts_cs1[t,0]=max(0, ghosts_cs1[t-1,0]-ppl_who_age[0]-all_cause_mort[0]*ghosts_cs1[t-1,0])
        ghosts_cs1[t,1]=max(0, ghosts_cs1[t-1,1]-ppl_who_age[1]+ppl_who_age[0]-all_cause_mort[1]*ghosts_cs1[t-1,1])
        ghosts_cs1[t,2]=max(0, ghosts_cs1[t-1,2]-ppl_who_age[2]+ppl_who_age[1]-all_cause_mort[2]*ghosts_cs1[t-1,2])
        ghosts_cs1[t,3]=max(0, ghosts_cs1[t-1,3]-ppl_who_age[3]+ppl_who_age[2]-all_cause_mort[3]*ghosts_cs1[t-1,3])
        ghosts_cs1[t,4]=max(0, ghosts_cs1[t-1,4]+ppl_who_age[3]-all_cause_mort[4]*ghosts_cs1[t-1,4])
         
        ghosts_cs1[t,:]= ghosts_cs1[t,:]+cs1_deaths[t,1]*age_of_deaths
        
    #DALYs
    bl_yll=np.sum(ghosts_cbl[:,0:5], axis=1)*discount[:,1]
    s1_yll=np.sum(ghosts_cs1[:,0:5], axis=1)*discount[:,1]
    
    bl_dalys=bl_yll+psens_bl["yld"]
    s1_dalys=s1_yll+psens_s1["yld"]
    
    #Productivity
    bl_prod=np.sum(ghosts_cbl[:,0:4], axis=1)*discount[:,1]*etp_ratio*gdp_grw[:,2]
    s1_prod=np.sum(ghosts_cs1[:,0:4], axis=1)*discount[:,1]*etp_ratio*gdp_grw[:,2]
    
    ## NEB
    bl_tc_ins=bl_prod[:]+dmc_bl[:,0]+imc_bl[:,0]
    s1_tc_ins=s1_prod[:]+dmc_s1[:,0]+imc_s1[:,0]
    
    bl_cum_tc=np.zeros(np.shape(bl_tc_ins))
    s1_cum_tc=np.zeros(np.shape(s1_tc_ins))
    
    for i in range(len(bl_tc_ins)):
        if i < 1:
            bl_cum_tc[i]=bl_tc_ins[i]
            s1_cum_tc[i]=s1_tc_ins[i]
        else:
            bl_cum_tc[i]=bl_cum_tc[i-1]+bl_tc_ins[i]
            s1_cum_tc[i]=s1_cum_tc[i-1]+s1_tc_ins[i]
    
    s1_neb=bl_cum_tc-s1_cum_tc
    ## FIX: sanitize NEB (NaN can propagate from Atomica params)
    s1_neb=np.nan_to_num(s1_neb, nan=0.0, posinf=0.0, neginf=0.0)
    
    ## ICER
    bl_dc_ins=dmc_bl[:,0]+imc_bl[:,0]
    s1_dc_ins=dmc_s1[:,0]+imc_s1[:,0]
    
    bl_cum_dc=np.zeros(np.shape(bl_dc_ins))
    s1_cum_dc=np.zeros(np.shape(s1_dc_ins))
    bl_cum_daly=np.zeros(np.shape(bl_dc_ins))
    s1_cum_daly=np.zeros(np.shape(s1_dc_ins))
    
    for i in range(len(bl_dc_ins)):
        if i <1:
            bl_cum_dc[i]=bl_dc_ins[i]
            bl_cum_daly[i]=bl_dalys[i]
            
            s1_cum_dc[i]=s1_dc_ins[i]
            s1_cum_daly[i]=s1_dalys[i]
        else:
            bl_cum_dc[i]=bl_dc_ins[i]+bl_cum_dc[i-1]
            bl_cum_daly[i]=bl_dalys[i]+bl_cum_daly[i-1]
            
            s1_cum_dc[i]=s1_dc_ins[i]+s1_cum_dc[i-1]
            s1_cum_daly[i]=s1_dalys[i]+s1_cum_daly[i-1]
        
    s1_icer=-(bl_cum_dc-s1_cum_dc)/(bl_cum_daly-s1_cum_daly)
    ## FIX: sanitize inf and nan (happens when DALY diff = 0 in early years)
    s1_icer=np.nan_to_num(s1_icer, nan=0.0, posinf=0.0, neginf=0.0)
    
    ## Return results needed for tabulation (within the input dictionaries for simplicity)
    
   
    psens_s1["cdaly_bl"]=bl_cum_daly
    psens_s1["cdaly_s1"]=s1_cum_daly
    
    psens_s1["cdirc_bl"]=bl_cum_dc
    psens_s1["cdirc_s1"]=s1_cum_dc
    
    psens_s1["prod"]=s1_prod
    psens_s1["icer"]=s1_icer
    psens_s1["neb"]=s1_neb
    
    return psens_s1


## MTCT worst case
afr_mtctw_s1=parsens_econ(afr_mtctw_bl, afr_mtctw_s1, "AFR", wd, "cost and agg data/costs.xlsx", 0.015)
amr_mtctw_s1=parsens_econ(amr_mtctw_bl, amr_mtctw_s1, "AMR", wd, "cost and agg data/costs.xlsx", 0.015)
emr_mtctw_s1=parsens_econ(emr_mtctw_bl, emr_mtctw_s1, "EMR", wd, "cost and agg data/costs.xlsx", 0.015)
eur_mtctw_s1=parsens_econ(eur_mtctw_bl, eur_mtctw_s1, "EUR", wd, "cost and agg data/costs.xlsx", 0.015)
sear_mtctw_s1=parsens_econ(sear_mtctw_bl, sear_mtctw_s1, "SEAR", wd, "cost and agg data/costs.xlsx", 0.015)
wpr_mtctw_s1=parsens_econ(wpr_mtctw_bl, wpr_mtctw_s1, "WPR", wd, "cost and agg data/costs.xlsx", 0.015)

## MTCT best case
afr_mtctb_s1=parsens_econ(afr_mtctb_bl, afr_mtctb_s1, "AFR", wd, "cost and agg data/costs.xlsx", 0.015)
amr_mtctb_s1=parsens_econ(amr_mtctb_bl, amr_mtctb_s1, "AMR", wd, "cost and agg data/costs.xlsx", 0.015)
emr_mtctb_s1=parsens_econ(emr_mtctb_bl, emr_mtctb_s1, "EMR", wd, "cost and agg data/costs.xlsx", 0.015)
eur_mtctb_s1=parsens_econ(eur_mtctb_bl, eur_mtctb_s1, "EUR", wd, "cost and agg data/costs.xlsx", 0.015)
sear_mtctb_s1=parsens_econ(sear_mtctb_bl, sear_mtctb_s1, "SEAR", wd, "cost and agg data/costs.xlsx", 0.015)
wpr_mtctb_s1=parsens_econ(wpr_mtctb_bl, wpr_mtctb_s1, "WPR", wd, "cost and agg data/costs.xlsx", 0.015)

## Mortality worst case
afr_mortw_s1=parsens_econ(afr_mortw_bl, afr_mortw_s1, "AFR", wd, "cost and agg data/costs.xlsx", 0.015)
amr_mortw_s1=parsens_econ(amr_mortw_bl, amr_mortw_s1, "AMR", wd, "cost and agg data/costs.xlsx", 0.015)
emr_mortw_s1=parsens_econ(emr_mortw_bl, emr_mortw_s1, "EMR", wd, "cost and agg data/costs.xlsx", 0.015)
eur_mortw_s1=parsens_econ(eur_mortw_bl, eur_mortw_s1, "EUR", wd, "cost and agg data/costs.xlsx", 0.015)
sear_mortw_s1=parsens_econ(sear_mortw_bl, sear_mortw_s1, "SEAR", wd, "cost and agg data/costs.xlsx", 0.015)
wpr_mortw_s1=parsens_econ(wpr_mortw_bl, wpr_mortw_s1, "WPR", wd, "cost and agg data/costs.xlsx", 0.015)

## Mortality best case
afr_mortb_s1=parsens_econ(afr_mortb_bl, afr_mortb_s1, "AFR", wd, "cost and agg data/costs.xlsx", 0.015)
amr_mortb_s1=parsens_econ(amr_mortb_bl, amr_mortb_s1, "AMR", wd, "cost and agg data/costs.xlsx", 0.015)
emr_mortb_s1=parsens_econ(emr_mortb_bl, emr_mortb_s1, "EMR", wd, "cost and agg data/costs.xlsx", 0.015)
eur_mortb_s1=parsens_econ(eur_mortb_bl, eur_mortb_s1, "EUR", wd, "cost and agg data/costs.xlsx", 0.015)
sear_mortb_s1=parsens_econ(sear_mortb_bl, sear_mortb_s1, "SEAR", wd, "cost and agg data/costs.xlsx", 0.015)
wpr_mortb_s1=parsens_econ(wpr_mortb_bl, wpr_mortb_s1, "WPR", wd, "cost and agg data/costs.xlsx", 0.015)

## Treatment worst case
afr_trtw_s1=parsens_econ(afr_trtw_bl, afr_trtw_s1, "AFR", wd, "cost and agg data/costs.xlsx", 0.015)
amr_trtw_s1=parsens_econ(amr_trtw_bl, amr_trtw_s1, "AMR", wd, "cost and agg data/costs.xlsx", 0.015)
emr_trtw_s1=parsens_econ(emr_trtw_bl, emr_trtw_s1, "EMR", wd, "cost and agg data/costs.xlsx", 0.015)
eur_trtw_s1=parsens_econ(eur_trtw_bl, eur_trtw_s1, "EUR", wd, "cost and agg data/costs.xlsx", 0.015)
sear_trtw_s1=parsens_econ(sear_trtw_bl, sear_trtw_s1, "SEAR", wd, "cost and agg data/costs.xlsx", 0.015)
wpr_trtw_s1=parsens_econ(wpr_trtw_bl, wpr_trtw_s1, "WPR", wd, "cost and agg data/costs.xlsx", 0.015)

## Treatment best case
afr_trtb_s1=parsens_econ(afr_trtb_bl, afr_trtb_s1, "AFR", wd, "cost and agg data/costs.xlsx", 0.015)
amr_trtb_s1=parsens_econ(amr_trtb_bl, amr_trtb_s1, "AMR", wd, "cost and agg data/costs.xlsx", 0.015)
emr_trtb_s1=parsens_econ(emr_trtb_bl, emr_trtb_s1, "EMR", wd, "cost and agg data/costs.xlsx", 0.015)
eur_trtb_s1=parsens_econ(eur_trtb_bl, eur_trtb_s1, "EUR", wd, "cost and agg data/costs.xlsx", 0.015)
sear_trtb_s1=parsens_econ(sear_trtb_bl, sear_trtb_s1, "SEAR", wd, "cost and agg data/costs.xlsx", 0.015)
wpr_trtb_s1=parsens_econ(wpr_trtb_bl, wpr_trtb_s1, "WPR", wd, "cost and agg data/costs.xlsx", 0.015)


## Global Outcomes for parameter scenarios
global_mtctw_s1={}
global_mtctb_s1={}
global_mortw_s1={}
global_mortb_s1={}
global_trtw_s1={}
global_trtb_s1={}

for key, val in afr_mtctw_s1.items():
    global_mtctw_s1[key]=afr_mtctw_s1[key]+amr_mtctw_s1[key]+emr_mtctw_s1[key]+eur_mtctw_s1[key]+sear_mtctw_s1[key]+wpr_mtctw_s1[key]
    global_mtctb_s1[key]=afr_mtctb_s1[key]+amr_mtctb_s1[key]+emr_mtctb_s1[key]+eur_mtctb_s1[key]+sear_mtctb_s1[key]+wpr_mtctb_s1[key]
    global_mortw_s1[key]=afr_mortw_s1[key]+amr_mortw_s1[key]+emr_mortw_s1[key]+eur_mortw_s1[key]+sear_mortw_s1[key]+wpr_mortw_s1[key]
    global_mortb_s1[key]=afr_mortb_s1[key]+amr_mortb_s1[key]+emr_mortb_s1[key]+eur_mortb_s1[key]+sear_mortb_s1[key]+wpr_mortb_s1[key]
    global_trtw_s1[key]=afr_trtw_s1[key]+amr_trtw_s1[key]+emr_trtw_s1[key]+eur_trtw_s1[key]+sear_trtw_s1[key]+wpr_trtw_s1[key]
    global_trtb_s1[key]=afr_trtb_s1[key]+amr_trtb_s1[key]+emr_trtb_s1[key]+eur_trtb_s1[key]+sear_trtb_s1[key]+wpr_trtb_s1[key]

global_mtctw_s1["icer"]=-(global_mtctw_s1["cdirc_bl"]-global_mtctw_s1["cdirc_s1"])/(global_mtctw_s1["cdaly_bl"]-global_mtctw_s1["cdaly_s1"])
global_mtctb_s1["icer"]=-(global_mtctb_s1["cdirc_bl"]-global_mtctb_s1["cdirc_s1"])/(global_mtctb_s1["cdaly_bl"]-global_mtctb_s1["cdaly_s1"])
global_mortw_s1["icer"]=-(global_mortw_s1["cdirc_bl"]-global_mortw_s1["cdirc_s1"])/(global_mortw_s1["cdaly_bl"]-global_mortw_s1["cdaly_s1"])
global_mortb_s1["icer"]=-(global_mortb_s1["cdirc_bl"]-global_mortb_s1["cdirc_s1"])/(global_mortb_s1["cdaly_bl"]-global_mortb_s1["cdaly_s1"])
global_trtw_s1["icer"]=-(global_trtw_s1["cdirc_bl"]-global_trtw_s1["cdirc_s1"])/(global_trtw_s1["cdaly_bl"]-global_trtw_s1["cdaly_s1"])
global_trtb_s1["icer"]=-(global_trtb_s1["cdirc_bl"]-global_trtb_s1["cdirc_s1"])/(global_trtb_s1["cdaly_bl"]-global_trtb_s1["cdaly_s1"])


def econsens_econ(bl_cent, cent_s1, econ, reg, cost_data, disc_rate):
    import numpy as np
    import pandas as pd
    
    #Discounting Array
    if disc_rate >1:
        disc_rate=disc_rate/100
    else:
        disc_rate=disc_rate
        
    discount=np.zeros((len(np.arange(1990,2100,1)),3))
    discount[:,0]=np.arange(1990,2100,1)
    
    for idx,val in enumerate(discount[:,0]):
        if val <= 2025:
            discount[idx,1]=1
            discount[idx,2]=0 #use this for high risk screening as a check
        else:
            discount[idx,1:]=(1+disc_rate)**(2025-val)
    
    vax_costs=pd.read_excel(cost_data, sheet_name="vax")
    care_costs=pd.read_excel(cost_data, sheet_name="care")

    
    """Vaccine Costs (just add/substract costs from relevant parts)"""
    #Current
    bd_pe=vax_costs[reg].iloc[0]
    hb3_pe=vax_costs[reg].iloc[1]
    
    bl_bd_pe, s1_bd_pe, bl_hb3_pe, s1_hb3_pe=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    bl_bd_pe[:,0]=bl_cent["bd_cov"][:]*bl_cent["births"][:]*bd_pe*discount[:,1]
    bl_hb3_pe[:,0]=bl_cent["hb3_cov"][:]*bl_cent["births"][:]*hb3_pe*discount[:,1]
    s1_bd_pe[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_pe*discount[:,1]
    s1_hb3_pe[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_pe*discount[:,1]
    
    #Lower Bound
    bd_lb=vax_costs[reg].iloc[6]
    hb3_lb=vax_costs[reg].iloc[8]
    
    bl_bd_lb, s1_bd_lb, bl_hb3_lb, s1_hb3_lb=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    bl_bd_lb[:,0]=bl_cent["bd_cov"][:]*bl_cent["births"][:]*bd_lb*discount[:,1]
    bl_hb3_lb[:,0]=bl_cent["hb3_cov"][:]*bl_cent["births"][:]*hb3_lb*discount[:,1]
    s1_bd_lb[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_lb*discount[:,1]
    s1_hb3_lb[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_lb*discount[:,1]
        
    #Upper Bound
    bd_ub=vax_costs[reg].iloc[7]
    hb3_ub=vax_costs[reg].iloc[9]
    
    bl_bd_ub, s1_bd_ub, bl_hb3_ub, s1_hb3_ub=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    bl_bd_ub[:,0]=bl_cent["bd_cov"][:]*bl_cent["births"][:]*bd_ub*discount[:,1]
    bl_hb3_ub[:,0]=bl_cent["hb3_cov"][:]*bl_cent["births"][:]*hb3_ub*discount[:,1]
    s1_bd_ub[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_ub*discount[:,1]
    s1_hb3_ub[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_ub*discount[:,1]
    
    # Difference (to be added to direct costs)
    
    bl_bd_lbd=bl_bd_lb-bl_bd_pe
    bl_hb3_lbd=bl_hb3_lb-bl_hb3_pe
    s1_bd_lbd=s1_bd_lb-s1_bd_pe
    s1_hb3_lbd=s1_hb3_lb-s1_hb3_pe
    
    bl_bd_ubd=bl_bd_ub-bl_bd_pe
    bl_hb3_ubd=bl_hb3_ub-bl_hb3_pe
    s1_bd_ubd=s1_bd_ub-s1_bd_pe
    s1_hb3_ubd=s1_hb3_ub-s1_hb3_pe
    
    #total direct costs for tabulation, NEB, and ICER
    vcost_lb_bl=econ["cbl_dirc"]+(bl_bd_lbd[:,0]+bl_hb3_lbd[:,0])
    vcost_ub_bl=econ["cbl_dirc"]+(bl_bd_ubd[:,0]+bl_hb3_ubd[:,0])
    
    vcost_lb_s1=econ["cs1_dirc"]+(s1_bd_lbd[:,0]+s1_hb3_lbd[:,0])
    vcost_ub_s1=econ["cs1_dirc"]+(s1_bd_ubd[:,0]+s1_hb3_ubd[:,0])
    
    """Treatment Costs (add mAVs)"""
    #current
    tx_pe=care_costs[reg].iloc[3]
    mav_cost=vax_costs[reg].iloc[11]
    
    tx_bl_pe=bl_cent["tx_cov"]*tx_pe*discount[:,1]
    mav_bl_s1=cent_s1["mav_n"][:]*mav_cost*discount[:,1]
    tx_s1_pe=cent_s1["tx_cov"]*tx_pe*discount[:,1]
    
    #halved
    tx_bl_lb=tx_bl_pe*0.5
    mav_s1_lb=mav_bl_s1*0.5
    tx_s1_lb=tx_s1_pe*0.5
    
    #doubled
    tx_bl_ub=tx_bl_pe*2
    mav_s1_ub=mav_bl_s1*2
    tx_s1_ub=tx_s1_pe*2
    
    # Difference (to be added to direct costs)
    tx_bl_lbd=tx_bl_lb-tx_bl_pe
    tx_bl_ubd=tx_bl_ub-tx_bl_pe
    
    tx_s1_lbd=(tx_s1_lb+mav_s1_lb)-(tx_s1_pe+mav_bl_s1)
    tx_s1_ubd=(tx_s1_ub+mav_s1_ub)-(tx_s1_pe+mav_bl_s1)
    
    #total direct costs for tabulation, NEB, and ICER
    tcost_lb_bl=econ["cbl_dirc"]+tx_bl_lbd
    tcost_ub_bl=econ["cbl_dirc"]+tx_bl_ubd
    
    tcost_lb_s1=econ["cs1_dirc"]+tx_s1_lbd
    tcost_ub_s1=econ["cs1_dirc"]+tx_s1_ubd
    
    """mAV with HBIG"""
    hrs_cost=vax_costs[reg].iloc[10]
    dx_cost=care_costs[reg].iloc[0]
    mav_hbig=vax_costs[reg].iloc[12]
    
    mv_bl_pe, mv_s1_pe, mv_bl_hbig, mv_s1_hbig=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)) 
    #current (no HBIG)
    mv_bl_pe[:,0]=(bl_cent["prg_scr"][:]*dx_cost*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    mv_s1_pe[:,0]=(cent_s1["mav_n"][:]*mav_cost*discount[:,1])+(cent_s1["prg_scr"][:]*dx_cost*discount[:,1])+(cent_s1["prg_hrs"][:]*hrs_cost*discount[:,2])
    # upper bound (with HBIG)
    mv_bl_hbig[:,0]=(bl_cent["prg_scr"][:]*dx_cost*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    mv_s1_hbig[:,0]=(cent_s1["mav_n"][:]*mav_hbig*discount[:,1])+(cent_s1["prg_scr"][:]*dx_cost*discount[:,1])+(cent_s1["prg_hrs"][:]*hrs_cost*discount[:,2])
    # Difference (to be added to direct costs)
    bl_mvd=mv_bl_hbig-mv_bl_pe
    s1_mvd=mv_s1_hbig-mv_s1_pe
    
    #total direct costs for tabulation, NEB, and ICER
    hbig_bl=econ["cbl_dirc"]+bl_mvd[:,0]
    hbig_s1=econ["cs1_dirc"]+s1_mvd[:,0]
    
    """HCC surveillance costs"""
    #current
    hcc_cost=care_costs[reg].iloc[6] #HCC surveillance costs
    hcc_prp=care_costs[reg].iloc[8] #proportion of CHB in 15-49y who are 40-49y (HCC screening)
    util=0.25
    
    hccs_bl_pe, hccs_s1_pe, hccs_bl_lb, hccs_s1_lb, hccs_bl_ub, hccs_s1_ub=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    hccs_bl_pe[:, 0]=((bl_cent["pop_hcc"][:]*hcc_cost*util)+(bl_cent["tgt_hcc"][:]*hcc_cost*util)+(bl_cent["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    hccs_s1_pe[:, 0]=((cent_s1["pop_hcc"][:]*hcc_cost*util)+(cent_s1["tgt_hcc"][:]*hcc_cost*util)+(cent_s1["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]

    #halved
    hccs_bl_lb=hccs_bl_pe*0.5
    hccs_s1_lb=hccs_s1_pe*0.5
    
    #doubled
    hccs_bl_ub=hccs_bl_pe*2
    hccs_s1_ub=hccs_s1_pe*2

    # Difference (to be added to indirect costs)
    bl_hccs_lbd=hccs_bl_lb-hccs_bl_pe
    bl_hccs_ubd=hccs_bl_ub-hccs_bl_pe
    
    s1_hccs_lbd=hccs_s1_lb-hccs_s1_pe
    s1_hccs_ubd=hccs_s1_ub-hccs_s1_pe
    
    #total direct costs for tabulation, NEB, and ICER
    hccs_lb_bl=econ["cbl_dirc"]+bl_hccs_lbd[:,0]
    hccs_ub_bl=econ["cbl_dirc"]+bl_hccs_ubd[:,0]
    
    hccs_lb_s1=econ["cs1_dirc"]+s1_hccs_lbd[:,0]
    hccs_ub_s1=econ["cs1_dirc"]+s1_hccs_ubd[:,0]
    
    """Hospitalization costs"""
    #current
    hosp_cost=care_costs[reg].iloc[7]
    tx_hosp=0.5
    
    hospc_bl_pe, hospc_s1_pe, hospc_bl_lb, hospc_s1_lb, hospc_bl_ub, hospc_s1_ub=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    hospc_bl_pe[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util)+(bl_cent["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    hospc_s1_pe[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util)+(cent_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    
    #halved
    hospc_bl_lb=hospc_bl_pe*0.5
    hospc_s1_lb=hospc_s1_pe*0.5
    
    #doubled
    hospc_bl_ub=hospc_bl_pe*2
    hospc_s1_ub=hospc_s1_pe*2
        
    # Difference (to be added to indirect costs)
    bl_hospc_lbd=hospc_bl_lb-hospc_bl_pe
    bl_hospc_ubd=hospc_bl_ub-hospc_bl_pe
    
    s1_hospc_lbd=hospc_s1_lb-hospc_s1_pe
    s1_hospc_ubd=hospc_s1_ub-hospc_s1_pe
    
    #total direct costs for tabulation, NEB, and ICER
    hospc_lb_bl=econ["cbl_dirc"]+bl_hospc_lbd[:,0]
    hospc_ub_bl=econ["cbl_dirc"]+bl_hospc_ubd[:,0]
    
    hospc_lb_s1=econ["cs1_dirc"]+s1_hospc_lbd[:,0]
    hospc_ub_s1=econ["cs1_dirc"]+s1_hospc_ubd[:,0]
    
    """UHC assumptions (vaccination, diagnosis and high-risk screening)"""
    bd_lb_hr=vax_costs[reg].iloc[2]
    hb3_lb_hr=vax_costs[reg].iloc[3]
    dx_lb_hr=care_costs[reg].iloc[1]
    hrs_lb_hr=vax_costs[reg].iloc[13]
    
    bd_ub_hr=vax_costs[reg].iloc[4]
    hb3_ub_hr=vax_costs[reg].iloc[5]
    dx_ub_hr=care_costs[reg].iloc[2]
    hrs_ub_hr=vax_costs[reg].iloc[14]
    
    #current
    bl_bd_pe, s1_bd_pe, bl_hb3_pe, s1_hb3_pe=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    bl_bd_pe[:,0]=bl_cent["bd_cov"][:]*bl_cent["births"][:]*bd_pe*discount[:,1]
    bl_hb3_pe[:,0]=bl_cent["hb3_cov"][:]*bl_cent["births"][:]*hb3_pe*discount[:,1]
    s1_bd_pe[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_pe*discount[:,1]
    s1_hb3_pe[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_pe*discount[:,1]
    
    mv_bl_pe, mv_s1_pe=np.zeros((110,1)),np.zeros((110,1))
    
    mv_bl_pe[:,0]=(bl_cent["prg_scr"][:]*dx_cost*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    mv_s1_pe[:,0]=(cent_s1["mav_n"][:]*mav_cost*discount[:,1])+(cent_s1["prg_scr"][:]*dx_cost*discount[:,1])+(cent_s1["prg_hrs"][:]*hrs_cost*discount[:,2])
    
    bl_dx_inc, s1_dx_inc=np.zeros((110,1)),np.zeros((110,1))
    
    for i in range(len(bl_cent["dx_prop"])):
        if i < 1:
            bl_dx_inc[i,0]=bl_cent["dx_prop"][i]
            s1_dx_inc[i,0]=cent_s1["dx_prop"][i]
        else:
            bl_dx_inc[i,0]=max(bl_cent["dx_prop"][i]-bl_cent["dx_prop"][i-1],0)
            s1_dx_inc[i,0]=max(cent_s1["dx_prop"][i]-cent_s1["dx_prop"][i-1],0)
    
    bl_dxc_pe, s1_dxc_pe=np.zeros((110,1)),np.zeros((110,1))
    
    for yr in range(len(bl_dx_inc)):
        bl_dxc_pe[yr,0]=(bl_cent["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*bl_dx_inc[yr,0]*(bl_cent["pop"][yr]-bl_cent["dx_rate"][yr])*discount[yr,1])
        s1_dxc_pe[yr,0]=(cent_s1["dx_rate"][yr]*dx_cost*discount[yr,1])+(dx_cost*s1_dx_inc[yr,0]*(cent_s1["pop"][yr]-cent_s1["dx_rate"][yr])*discount[yr,1])
        
    #0%
    bl_bd_ulb, s1_bd_ulb, bl_hb3_ulb, s1_hb3_ulb=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    bl_bd_ulb[:,0]=bl_cent["bd_cov"][:]*bl_cent["births"][:]*bd_lb_hr*discount[:,1]
    bl_hb3_ulb[:,0]=bl_cent["hb3_cov"][:]*bl_cent["births"][:]*hb3_lb_hr*discount[:,1]
    s1_bd_ulb[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_lb_hr*discount[:,1]
    s1_hb3_ulb[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_lb_hr*discount[:,1]
    
    mv_bl_ulb, mv_s1_ulb=np.zeros((110,1)),np.zeros((110,1))
    
    mv_bl_ulb[:,0]=(bl_cent["prg_scr"][:]*dx_lb_hr*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    mv_s1_ulb[:,0]=(cent_s1["mav_n"][:]*mav_cost*discount[:,1])+(cent_s1["prg_scr"][:]*dx_lb_hr*discount[:,1])+(cent_s1["prg_hrs"][:]*hrs_lb_hr*discount[:,2])
    
    bl_dxc_ulb, s1_dxc_ulb=np.zeros((110,1)),np.zeros((110,1))
    
    for yr in range(len(bl_dx_inc)):
        bl_dxc_ulb[yr,0]=(bl_cent["dx_rate"][yr]*dx_lb_hr*discount[yr,1])+(dx_lb_hr*bl_dx_inc[yr,0]*(bl_cent["pop"][yr]-bl_cent["dx_rate"][yr])*discount[yr,1])
        s1_dxc_ulb[yr,0]=(cent_s1["dx_rate"][yr]*dx_lb_hr*discount[yr,1])+(dx_lb_hr*s1_dx_inc[yr,0]*(cent_s1["pop"][yr]-cent_s1["dx_rate"][yr])*discount[yr,1])
        
    #100%
    bl_bd_uub, s1_bd_uub, bl_hb3_uub, s1_hb3_uub=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))
    
    bl_bd_uub[:,0]=bl_cent["bd_cov"][:]*bl_cent["births"][:]*bd_ub_hr*discount[:,1]
    bl_hb3_uub[:,0]=bl_cent["hb3_cov"][:]*bl_cent["births"][:]*hb3_ub_hr*discount[:,1]
    s1_bd_uub[:,0]=cent_s1["bd_cov"][:]*cent_s1["births"][:]*bd_ub_hr*discount[:,1]
    s1_hb3_uub[:,0]=cent_s1["hb3_cov"][:]*cent_s1["births"][:]*hb3_ub_hr*discount[:,1]
    
    mv_bl_uub, mv_s1_uub=np.zeros((110,1)),np.zeros((110,1))
    
    mv_bl_uub[:,0]=(bl_cent["prg_scr"][:]*dx_ub_hr*discount[:,1])#(cent["mav_n"][:]*mav_cost*discount[:,1])+(cent["prg_hrs"][:]*hrs_cost*discount[:,1])
    mv_s1_uub[:,0]=(cent_s1["mav_n"][:]*mav_cost*discount[:,1])+(cent_s1["prg_scr"][:]*dx_ub_hr*discount[:,1])+(cent_s1["prg_hrs"][:]*hrs_ub_hr*discount[:,2])
    
    bl_dxc_uub, s1_dxc_uub=np.zeros((110,1)),np.zeros((110,1))
    
    for yr in range(len(bl_dx_inc)):
        bl_dxc_uub[yr,0]=(bl_cent["dx_rate"][yr]*dx_ub_hr*discount[yr,1])+(dx_ub_hr*bl_dx_inc[yr,0]*(bl_cent["pop"][yr]-bl_cent["dx_rate"][yr])*discount[yr,1])
        s1_dxc_uub[yr,0]=(cent_s1["dx_rate"][yr]*dx_ub_hr*discount[yr,1])+(dx_ub_hr*s1_dx_inc[yr,0]*(cent_s1["pop"][yr]-cent_s1["dx_rate"][yr])*discount[yr,1])
        
    # Difference (to be added to direct (dx, mav, hepB-bd) costs)
    bl_lb_uhcd=(bl_bd_ulb+bl_hb3_ulb+mv_bl_ulb+bl_dxc_ulb)-(bl_bd_pe+bl_hb3_pe+mv_bl_pe+bl_dxc_pe)
    bl_ub_uhcd=(bl_bd_uub+bl_hb3_uub+mv_bl_uub+bl_dxc_uub)-(bl_bd_pe+bl_hb3_pe+mv_bl_pe+bl_dxc_pe)
    
    s1_lb_uhcd=(s1_bd_ulb+s1_hb3_ulb+mv_s1_ulb+s1_dxc_ulb)-(s1_bd_pe+s1_hb3_pe+mv_s1_pe+s1_dxc_pe)
    s1_ub_uhcd=(s1_bd_uub+s1_hb3_uub+mv_s1_uub+s1_dxc_uub)-(s1_bd_pe+s1_hb3_pe+mv_s1_pe+s1_dxc_pe)
    
    #total direct costs for tabulation, NEB, and ICER
    uhc_lb_bl=econ["cbl_dirc"]+bl_lb_uhcd[:,0]
    uhc_ub_bl=econ["cbl_dirc"]+bl_ub_uhcd[:,0]
    
    uhc_lb_s1=econ["cs1_dirc"]+s1_lb_uhcd[:,0]
    uhc_ub_s1=econ["cs1_dirc"]+s1_ub_uhcd[:,0]
    
    """Utilization assumptions (HCC surveillance and hospitalisation)"""
    util_lb=0
    util_ub=0.5
    
    dx_dmc=care_costs[reg].iloc[4] #cost of disease management (diagnosed)
    tx_dmc=care_costs[reg].iloc[4] #cost of disease management (on treatment)
    
    #current
    
    manu_bl_pe, manu_s1_pe=np.zeros((110,1)), np.zeros((110,1))
    
    manu_bl_pe[:,0]=((bl_cent["dm_dx"][:]*dx_dmc*util)+(bl_cent["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    manu_s1_pe[:,0]=((cent_s1["dm_dx"][:]*dx_dmc*util)+(cent_s1["dm_tx"][:]*tx_dmc*util))*discount[:,1]
    
    
    hospu_bl_pe, hospu_s1_pe=np.zeros((110,1)),np.zeros((110,1))
    
    hospu_bl_pe[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util)+(bl_cent["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    hospu_s1_pe[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util)+(cent_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    
    hccu_bl_pe, hccu_s1_pe=np.zeros((110,1)),np.zeros((110,1))
    
    hccu_bl_pe[:, 0]=((bl_cent["pop_hcc"][:]*hcc_cost*util)+(bl_cent["tgt_hcc"][:]*hcc_cost*util)+(bl_cent["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    hccu_s1_pe[:, 0]=((cent_s1["pop_hcc"][:]*hcc_cost*util)+(cent_s1["tgt_hcc"][:]*hcc_cost*util)+(cent_s1["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util))*discount[:,1]
    
    #0%
    manu_bl_lb, manu_s1_lb=np.zeros((110,1)), np.zeros((110,1))
    
    manu_bl_lb[:,0]=((bl_cent["dm_dx"][:]*dx_dmc*util_lb)+(bl_cent["dm_tx"][:]*tx_dmc*util_lb))*discount[:,1]
    manu_s1_lb[:,0]=((cent_s1["dm_dx"][:]*dx_dmc*util_lb)+(cent_s1["dm_tx"][:]*tx_dmc*util_lb))*discount[:,1]
    
    hospu_bl_lb, hospu_s1_lb=np.zeros((110,1)),np.zeros((110,1))
    
    hospu_bl_lb[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util_lb)+(bl_cent["hsp_tx"][:]*hosp_cost*util_lb*tx_hosp))*discount[:,1]
    hospu_s1_lb[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util_lb)+(cent_s1["hsp_tx"][:]*hosp_cost*util_lb*tx_hosp))*discount[:,1]
    
    hccu_bl_lb, hccu_s1_lb=np.zeros((110,1)),np.zeros((110,1))
    
    hccu_bl_lb[:, 0]=((bl_cent["pop_hcc"][:]*hcc_cost*util_lb)+(bl_cent["tgt_hcc"][:]*hcc_cost*util_lb)+(bl_cent["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util_lb))*discount[:,1]
    hccu_s1_lb[:, 0]=((cent_s1["pop_hcc"][:]*hcc_cost*util_lb)+(cent_s1["tgt_hcc"][:]*hcc_cost*util_lb)+(cent_s1["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util_lb))*discount[:,1]
    
    #50%
    manu_bl_ub, manu_s1_ub=np.zeros((110,1)), np.zeros((110,1))
    
    manu_bl_ub[:,0]=((bl_cent["dm_dx"][:]*dx_dmc*util_ub)+(bl_cent["dm_tx"][:]*tx_dmc*util_ub))*discount[:,1]
    manu_s1_ub[:,0]=((cent_s1["dm_dx"][:]*dx_dmc*util_ub)+(cent_s1["dm_tx"][:]*tx_dmc*util_ub))*discount[:,1]
    
    
    hospu_bl_ub, hospu_s1_ub=np.zeros((110,1)),np.zeros((110,1))
    
    hospu_bl_ub[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util_ub)+(bl_cent["hsp_tx"][:]*hosp_cost*util_ub*tx_hosp))*discount[:,1]
    hospu_s1_ub[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util_ub)+(cent_s1["hsp_tx"][:]*hosp_cost*util_ub*tx_hosp))*discount[:,1]
    
    hccu_bl_ub, hccu_s1_ub=np.zeros((110,1)),np.zeros((110,1))
    
    hccu_bl_ub[:, 0]=((bl_cent["pop_hcc"][:]*hcc_cost*util_ub)+(bl_cent["tgt_hcc"][:]*hcc_cost*util_ub)+(bl_cent["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util_ub))*discount[:,1]
    hccu_s1_ub[:, 0]=((cent_s1["pop_hcc"][:]*hcc_cost*util_ub)+(cent_s1["tgt_hcc"][:]*hcc_cost*util_ub)+(cent_s1["tgt_hcc_b"][:]*hcc_prp*hcc_cost*util_ub))*discount[:,1]
    # Difference (to be added to indirect (hcc surveillance, dis man, hosp) costs)
    bl_util_lbd=(hospu_bl_lb+hccu_bl_lb+manu_bl_lb)-(hospu_bl_pe+hccu_bl_pe+manu_bl_pe)
    bl_util_ubd=(hospu_bl_ub+hccu_bl_ub+manu_bl_ub)-(hospu_bl_pe+hccu_bl_pe+manu_bl_pe)
    
    s1_util_lbd=(hospu_s1_lb+hccu_s1_lb+manu_s1_lb)-(hospu_s1_pe+hccu_s1_pe+manu_bl_pe)
    s1_util_ubd=(hospu_s1_ub+hccu_s1_ub+manu_s1_ub)-(hospu_s1_pe+hccu_s1_pe+manu_bl_pe)
    
    #total direct costs for tabulation, NEB, and ICER
    util_lb_bl=econ["cbl_dirc"]+bl_util_lbd[:,0]
    util_ub_bl=econ["cbl_dirc"]+bl_util_ubd[:,0]
    
    util_lb_s1=econ["cs1_dirc"]+s1_util_lbd[:,0]
    util_ub_s1=econ["cs1_dirc"]+s1_util_ubd[:,0]
    
    """Hospitalisation (tx) assumption"""
    tx_hosp_lb=1
    tx_hosp_ub=0.25
    
    hospt_bl_pe, hospt_s1_pe, hospt_bl_lb, hospt_s1_lb, hospt_bl_ub, hospt_s1_ub=np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1)),np.zeros((110,1))

    #current
    hospt_bl_pe[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util)+(bl_cent["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    hospt_s1_pe[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util)+(cent_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp))*discount[:,1]
    #0%
    hospt_bl_lb[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util)+(bl_cent["hsp_tx"][:]*hosp_cost*util*tx_hosp_lb))*discount[:,1]
    hospt_s1_lb[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util)+(cent_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp_lb))*discount[:,1]
    #75%
    hospt_bl_ub[:, 0]=((bl_cent["hsp_utx"][:]*hosp_cost*util)+(bl_cent["hsp_tx"][:]*hosp_cost*util*tx_hosp_ub))*discount[:,1]
    hospt_s1_ub[:, 0]=((cent_s1["hsp_utx"][:]*hosp_cost*util)+(cent_s1["hsp_tx"][:]*hosp_cost*util*tx_hosp_ub))*discount[:,1]
    
    # Difference (to be added to indirect costs)
    bl_hospt_lbd=hospt_bl_lb-hospt_bl_pe
    bl_hospt_ubd=hospt_bl_ub-hospt_bl_pe
    
    s1_hospt_lbd=hospt_s1_lb-hospt_s1_pe
    s1_hospt_ubd=hospt_s1_ub-hospt_s1_pe
    
    #total direct costs for tabulation, NEB, and ICER
    hospt_lb_bl=econ["cbl_dirc"]+bl_hospt_lbd[:,0]
    hospt_ub_bl=econ["cbl_dirc"]+bl_hospt_ubd[:,0]
    
    hospt_lb_s1=econ["cs1_dirc"]+s1_hospt_lbd[:,0]
    hospt_ub_s1=econ["cs1_dirc"]+s1_hospt_ubd[:,0]
    
    res_dict={}
    res_dict["vax_lb_bl"]=vcost_lb_bl
    res_dict["vax_ub_bl"]=vcost_ub_bl
    res_dict["vax_lb_s1"]=vcost_lb_s1
    res_dict["vax_ub_s1"]=vcost_ub_s1
    res_dict["trt_lb_bl"]=tcost_lb_bl
    res_dict["trt_ub_bl"]=tcost_ub_bl
    res_dict["trt_lb_s1"]=tcost_lb_s1
    res_dict["trt_ub_s1"]=tcost_ub_s1
    res_dict["hbig_bl"]=hbig_bl
    res_dict["hbig_s1"]=hbig_s1
    res_dict["surv_lb_bl"]=hccs_lb_bl
    res_dict["surv_ub_bl"]=hccs_ub_bl
    res_dict["surv_lb_s1"]=hccs_lb_s1
    res_dict["surv_ub_s1"]=hccs_ub_s1
    res_dict["hosp_lb_bl"]=hospc_lb_bl
    res_dict["hosp_ub_bl"]=hospc_ub_bl
    res_dict["hosp_lb_s1"]=hospc_lb_s1
    res_dict["hosp_ub_s1"]=hospc_ub_s1
    res_dict["uhc_lb_bl"]=uhc_lb_bl
    res_dict["uhc_ub_bl"]=uhc_ub_bl
    res_dict["uhc_lb_s1"]=uhc_lb_s1
    res_dict["uhc_ub_s1"]=uhc_ub_s1
    res_dict["util_lb_bl"]=util_lb_bl
    res_dict["util_ub_bl"]=util_ub_bl
    res_dict["util_lb_s1"]=util_lb_s1
    res_dict["util_ub_s1"]=util_ub_s1
    res_dict["htx_lb_bl"]=hospt_lb_bl
    res_dict["htx_ub_bl"]=hospt_ub_bl
    res_dict["htx_lb_s1"]=hospt_lb_s1
    res_dict["htx_ub_s1"]=hospt_ub_s1
    
    return res_dict
    
afr_econ_sens= econsens_econ(afr_bl_cent, afr_s1_cent, afr_econ, "AFR", "cost and agg data/costs.xlsx", 0.015)   
amr_econ_sens= econsens_econ(amr_bl_cent, amr_s1_cent, amr_econ, "AMR", "cost and agg data/costs.xlsx", 0.015)     
emr_econ_sens= econsens_econ(emr_bl_cent, emr_s1_cent, emr_econ, "EMR", "cost and agg data/costs.xlsx", 0.015) 
eur_econ_sens= econsens_econ(eur_bl_cent, eur_s1_cent, eur_econ, "EUR", "cost and agg data/costs.xlsx", 0.015) 
sear_econ_sens= econsens_econ(sear_bl_cent, sear_s1_cent, sear_econ, "SEAR", "cost and agg data/costs.xlsx", 0.015)   
wpr_econ_sens= econsens_econ(wpr_bl_cent, wpr_s1_cent, wpr_econ, "WPR", "cost and agg data/costs.xlsx", 0.015) 
    
## Global Outcomes for economic scenarios
global_econ_sens={}

for key, val in afr_econ_sens.items():
    global_econ_sens[key]=afr_econ_sens[key]+amr_econ_sens[key]+emr_econ_sens[key]+eur_econ_sens[key]+sear_econ_sens[key]+wpr_econ_sens[key]

def sens_nebicer(econ_sens, econ_main):
    
    import numpy as np
    econ_outs={}
   
    ## Vaccine Costs
    econ_outs["vax_lb_neb"]=(np.cumsum(econ_sens["vax_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["vax_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["vax_ub_neb"]=(np.cumsum(econ_sens["vax_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["vax_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["vax_lb_icer"]=-(np.cumsum(econ_sens["vax_lb_bl"])-np.cumsum(econ_sens["vax_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["vax_ub_icer"]=-(np.cumsum(econ_sens["vax_ub_bl"])-np.cumsum(econ_sens["vax_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

    ## Treatment Costs
    econ_outs["tc_lb_neb"]=(np.cumsum(econ_sens["trt_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["trt_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["tc_ub_neb"]=(np.cumsum(econ_sens["trt_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["trt_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["tc_lb_icer"]=-(np.cumsum(econ_sens["trt_lb_bl"])-np.cumsum(econ_sens["trt_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["tc_ub_icer"]=-(np.cumsum(econ_sens["trt_ub_bl"])-np.cumsum(econ_sens["trt_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    
    ##HBIG
    econ_outs["hbig_neb"]=(np.cumsum(econ_sens["hbig_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["hbig_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["hbig_icer"]=-(np.cumsum(econ_sens["hbig_bl"])-np.cumsum(econ_sens["hbig_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    
    #HCC Surveillance
    econ_outs["surv_lb_neb"]=(np.cumsum(econ_sens["surv_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["surv_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["surv_ub_neb"]=(np.cumsum(econ_sens["surv_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["surv_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["surv_lb_icer"]=-(np.cumsum(econ_sens["surv_lb_bl"])-np.cumsum(econ_sens["surv_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["surv_ub_icer"]=-(np.cumsum(econ_sens["surv_ub_bl"])-np.cumsum(econ_sens["surv_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    
    #Hospital Costs
    econ_outs["hosp_lb_neb"]=(np.cumsum(econ_sens["hosp_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["hosp_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["hosp_ub_neb"]=(np.cumsum(econ_sens["hosp_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["hosp_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["hosp_lb_icer"]=-(np.cumsum(econ_sens["hosp_lb_bl"])-np.cumsum(econ_sens["hosp_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["hosp_ub_icer"]=-(np.cumsum(econ_sens["hosp_ub_bl"])-np.cumsum(econ_sens["hosp_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    
    #UHC assumpyion
    econ_outs["uhc_lb_neb"]=(np.cumsum(econ_sens["uhc_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["uhc_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["uhc_ub_neb"]=(np.cumsum(econ_sens["uhc_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["uhc_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["uhc_lb_icer"]=-(np.cumsum(econ_sens["uhc_lb_bl"])-np.cumsum(econ_sens["uhc_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["uhc_ub_icer"]=-(np.cumsum(econ_sens["uhc_ub_bl"])-np.cumsum(econ_sens["uhc_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    
    #Utilization assumption
    econ_outs["util_lb_neb"]=(np.cumsum(econ_sens["util_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["util_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["util_ub_neb"]=(np.cumsum(econ_sens["util_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["util_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["util_lb_icer"]=-(np.cumsum(econ_sens["util_lb_bl"])-np.cumsum(econ_sens["util_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["util_ub_icer"]=-(np.cumsum(econ_sens["util_ub_bl"])-np.cumsum(econ_sens["util_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    
    #Treatment effectiveness (hospitalizations)
    econ_outs["htx_lb_neb"]=(np.cumsum(econ_sens["htx_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["htx_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["htx_ub_neb"]=(np.cumsum(econ_sens["htx_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["htx_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))
    
    econ_outs["htx_lb_icer"]=-(np.cumsum(econ_sens["htx_lb_bl"])-np.cumsum(econ_sens["htx_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["htx_ub_icer"]=-(np.cumsum(econ_sens["htx_ub_bl"])-np.cumsum(econ_sens["htx_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

    return econ_outs

afr_esens_outs=sens_nebicer(afr_econ_sens, afr_econ)
amr_esens_outs=sens_nebicer(amr_econ_sens, amr_econ)
emr_esens_outs=sens_nebicer(emr_econ_sens, emr_econ)
eur_esens_outs=sens_nebicer(eur_econ_sens, eur_econ)
sear_esens_outs=sens_nebicer(sear_econ_sens, sear_econ)
wpr_esens_outs=sens_nebicer(wpr_econ_sens, wpr_econ)
global_esens_outs=sens_nebicer(global_econ_sens, global_econ)

def sens_tables(s1_cent, econ, mtctw, mtctb, mortw, mortb, txtw, txtb, esens, esens_outs):
    import pandas as pd
    import numpy as np
    year=np.arange(1990, 2100, 1)
 
    
    table=pd.DataFrame(columns=["Scenario", "CHB Incidence (millions)", "HCC Incidence (millions)", "HBV attributable deaths (millions)", "DALYs (millions)",
                          "Healthcare Costs (US$ billions)", "Productivity Losses (US$, billions)", "ICER", "Net Economic Benefit (year achieved)"])
    
    table["Scenario"]=["Base Case", "MTCT worst case", "MTCT best case", "Mortality worst case", "Mortality best case", "Treatment Effectiveness worst case", 
                       "Treatment Effectiveness best case", "Lower bound vaccine commodity cost", "Upper bound vaccine commodity cost", "Lower bound TDF cost",
                       "Upper bound TDF cost", "HBIG + maternal antivirals", "Lower bound HCC surveillance cost", "Upper bound HCC surveillance cost", "Lower bound hospitalisation cost",
                       "Upper bound hospitalisation cost", "Lower bound UHC absorption", "Upper bound UHC absorption", "Lower bound care utilization", "Upper bound care utilization",
                       "Lower bound treatment hospital prevention", "Upper bound treatment hospital prevention"]
    
    ## Base Case
    table.iloc[0,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[0,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[0,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[0,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[0,5]=round(np.sum(econ["cs1_dirc"][36:110]/1e9),2)
    table.iloc[0,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[0,7]=round(econ["cs1_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(econ["cs1_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[0,8]=year[yneb_i]
    
    ## MTCT w
    table.iloc[1,1]=round(np.sum(mtctw["chb_inc"][36:110])/1e6,2)
    table.iloc[1,2]=round(np.sum(mtctw["hcc_inc"][36:110])/1e6,2)
    table.iloc[1,3]=round(np.sum(mtctw["mort"][36:110])/1e6,2)
    table.iloc[1,4]=round((mtctw["cdaly_s1"][109]-mtctw["cdaly_s1"][35])/1e6,2)
    table.iloc[1,5]=round((mtctw["cdirc_s1"][109]-mtctw["cdirc_s1"][35])/1e9,2)
    table.iloc[1,6]=round(np.sum(mtctw["prod"][36:110]/1e9),2)
    table.iloc[1,7]=round(mtctw["icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(mtctw["neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[1,8]=year[yneb_i]
    ## MTCT b
    table.iloc[2,1]=round(np.sum(mtctb["chb_inc"][36:110])/1e6,2)
    table.iloc[2,2]=round(np.sum(mtctb["hcc_inc"][36:110])/1e6,2)
    table.iloc[2,3]=round(np.sum(mtctb["mort"][36:110])/1e6,2)
    table.iloc[2,4]=round((mtctb["cdaly_s1"][109]-mtctb["cdaly_s1"][35])/1e6,2)
    table.iloc[2,5]=round((mtctb["cdirc_s1"][109]-mtctb["cdirc_s1"][35])/1e9,2)
    table.iloc[2,6]=round(np.sum(mtctb["prod"][36:110]/1e9),2)
    table.iloc[2,7]=round(mtctb["icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(mtctb["neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[2,8]=year[yneb_i]
    ## mort w
    table.iloc[3,1]=round(np.sum(mortw["chb_inc"][36:110])/1e6,2)
    table.iloc[3,2]=round(np.sum(mortw["hcc_inc"][36:110])/1e6,2)
    table.iloc[3,3]=round(np.sum(mortw["mort"][36:110])/1e6,2)
    table.iloc[3,4]=round((mortw["cdaly_s1"][109]-mortw["cdaly_s1"][35])/1e6,2)
    table.iloc[3,5]=round((mortw["cdirc_s1"][109]-mortw["cdirc_s1"][35])/1e9,2)
    table.iloc[3,6]=round(np.sum(mortw["prod"][36:110]/1e9),2)
    table.iloc[3,7]=round(mortw["icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(mortw["neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[3,8]=year[yneb_i]
    ## mort b
    table.iloc[4,1]=round(np.sum(mortb["chb_inc"][36:110])/1e6,2)
    table.iloc[4,2]=round(np.sum(mortb["hcc_inc"][36:110])/1e6,2)
    table.iloc[4,3]=round(np.sum(mortb["mort"][36:110])/1e6,2)
    table.iloc[4,4]=round((mortb["cdaly_s1"][109]-mortb["cdaly_s1"][35])/1e6,2)
    table.iloc[4,5]=round((mortb["cdirc_s1"][109]-mortb["cdirc_s1"][35])/1e9,2)
    table.iloc[4,6]=round(np.sum(mortb["prod"][36:110]/1e9),2)
    table.iloc[4,7]=round(mortb["icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(mortb["neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[4,8]=year[yneb_i]
    ## tx w
    table.iloc[5,1]=round(np.sum(txtw["chb_inc"][36:110])/1e6,2)
    table.iloc[5,2]=round(np.sum(txtw["hcc_inc"][36:110])/1e6,2)
    table.iloc[5,3]=round(np.sum(txtw["mort"][36:110])/1e6,2)
    table.iloc[5,4]=round((txtw["cdaly_s1"][109]-txtw["cdaly_s1"][35])/1e6,2)
    table.iloc[5,5]=round((txtw["cdirc_s1"][109]-txtw["cdirc_s1"][35])/1e9,2)
    table.iloc[5,6]=round(np.sum(txtw["prod"][36:110]/1e9),2)
    table.iloc[5,7]=round(txtw["icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(txtw["neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[5,8]=year[yneb_i]
    ##tx b
    table.iloc[6,1]=round(np.sum(txtb["chb_inc"][36:110])/1e6,2)
    table.iloc[6,2]=round(np.sum(txtb["hcc_inc"][36:110])/1e6,2)
    table.iloc[6,3]=round(np.sum(txtb["mort"][36:110])/1e6,2)
    table.iloc[6,4]=round((txtb["cdaly_s1"][109]-txtb["cdaly_s1"][35])/1e6,2)
    table.iloc[6,5]=round((txtb["cdirc_s1"][109]-txtb["cdirc_s1"][35])/1e9,2)
    table.iloc[6,6]=round(np.sum(txtb["prod"][36:110]/1e9),2)
    table.iloc[6,7]=round(txtb["icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(txtb["neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[6,8]=year[yneb_i]
  ##tx b
    ## vax com cost lb
    table.iloc[7,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[7,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[7,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[7,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[7,5]=round(np.sum(esens["vax_lb_s1"][36:110]/1e9),2)
    table.iloc[7,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[7,7]=round(esens_outs["vax_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["vax_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[7,8]=year[yneb_i]
    
    ## vax com cost ub
    table.iloc[8,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[8,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[8,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[8,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[8,5]=round(np.sum(esens["vax_ub_s1"][36:110]/1e9),2)
    table.iloc[8,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[8,7]=round(esens_outs["vax_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["vax_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[8,8]=year[yneb_i]
    # tdf cost lb
    table.iloc[9,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[9,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[9,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[9,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[9,5]=round(np.sum(esens["trt_lb_s1"][36:110]/1e9),2)
    table.iloc[9,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[9,7]=round(esens_outs["tc_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["tc_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[9,8]=year[yneb_i]
    
    # tdf cost ub
    table.iloc[10,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[10,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[10,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[10,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[10,5]=round(np.sum(esens["trt_ub_s1"][36:110]/1e9),2)
    table.iloc[10,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[10,7]=round(esens_outs["tc_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["tc_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[10,8]=year[yneb_i]
    
    
    # hbig
    table.iloc[11,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[11,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[11,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[11,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[11,5]=round(np.sum(esens["hbig_s1"][36:110]/1e9),2)
    table.iloc[11,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[11,7]=round(esens_outs["hbig_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["hbig_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[11,8]=year[yneb_i]
    # hcc surveillance lb
    table.iloc[12,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[12,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[12,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[12,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[12,5]=round(np.sum(esens["surv_lb_s1"][36:110]/1e9),2)
    table.iloc[12,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[12,7]=round(esens_outs["surv_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["surv_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[12,8]=year[yneb_i]
    # hcc surveillance ub
    table.iloc[13,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[13,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[13,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[13,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[13,5]=round(np.sum(esens["surv_ub_s1"][36:110]/1e9),2)
    table.iloc[13,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[13,7]=round(esens_outs["surv_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["surv_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[13,8]=year[yneb_i]
    #hosp cost lb
    table.iloc[14,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[14,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[14,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[14,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[14,5]=round(np.sum(esens["hosp_lb_s1"][36:110]/1e9),2)
    table.iloc[14,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[14,7]=round(esens_outs["hosp_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["hosp_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[14,8]=year[yneb_i]
    #hosp cost ub
    table.iloc[15,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[15,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[15,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[15,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[15,5]=round(np.sum(esens["hosp_ub_s1"][36:110]/1e9),2)
    table.iloc[15,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[15,7]=round(esens_outs["hosp_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["hosp_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[15,8]=year[yneb_i]
    #uhc lb
    table.iloc[16,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[16,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[16,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[16,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[16,5]=round(np.sum(esens["uhc_lb_s1"][36:110]/1e9),2)
    table.iloc[16,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[16,7]=round(esens_outs["uhc_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["uhc_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[16,8]=year[yneb_i]
    #uhc ub
    table.iloc[17,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[17,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[17,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[17,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[17,5]=round(np.sum(esens["uhc_ub_s1"][36:110]/1e9),2)
    table.iloc[17,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[17,7]=round(esens_outs["uhc_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["uhc_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[17,8]=year[yneb_i]
    #util lb
    table.iloc[18,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[18,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[18,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[18,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[18,5]=round(np.sum(esens["util_lb_s1"][36:110]/1e9),2)
    table.iloc[18,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[18,7]=round(esens_outs["util_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["util_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[18,8]=year[yneb_i]
    #util ub
    table.iloc[19,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[19,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[19,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[19,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[19,5]=round(np.sum(esens["util_ub_s1"][36:110]/1e9),2)
    table.iloc[19,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[19,7]=round(esens_outs["util_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["util_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[19,8]=year[yneb_i]
    #tx hosp lb
    table.iloc[20,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[20,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[20,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[20,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[20,5]=round(np.sum(esens["htx_lb_s1"][36:110]/1e9),2)
    table.iloc[20,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[20,7]=round(esens_outs["htx_lb_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["htx_lb_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[20,8]=year[yneb_i]
    #tx hosp ub
    table.iloc[21,1]=round(np.sum(s1_cent["chb_inc"][36:110])/1e6,2)
    table.iloc[21,2]=round(np.sum(s1_cent["hcc_inc"][36:110])/1e6,2)
    table.iloc[21,3]=round(np.sum(s1_cent["mort"][36:110])/1e6,2)
    table.iloc[21,4]=round(np.sum(econ["cs1_daly"][36:110]/1e6),2)
    table.iloc[21,5]=round(np.sum(esens["htx_ub_s1"][36:110]/1e9),2)
    table.iloc[21,6]=round(np.sum(econ["cs1_prod"][36:110]/1e9),2)
    table.iloc[21,7]=round(esens_outs["htx_ub_icer"][109],2)
    year_idx=[]
    for idx,val in enumerate(esens_outs["htx_ub_neb"]):
        if val>0 and year[idx] >= 2027:
            year_idx.append(idx)
    yneb_i=safe_min(year_idx)
    table.iloc[21,8]=year[yneb_i]
    
    return table

## Summary table of sensitivity analysis outcomes for each WHO region (and global) - can be exported to csv using pandas functionality if desired.
afr_sens_tab=sens_tables(afr_s1_cent, afr_econ, afr_mtctw_s1, afr_mtctb_s1, afr_mortw_s1, afr_mortb_s1, afr_trtw_s1, afr_trtb_s1, afr_econ_sens, afr_esens_outs)
amr_sens_tab=sens_tables(amr_s1_cent, amr_econ, amr_mtctw_s1, amr_mtctb_s1, amr_mortw_s1, amr_mortb_s1, amr_trtw_s1, amr_trtb_s1, amr_econ_sens, amr_esens_outs)
emr_sens_tab=sens_tables(emr_s1_cent, emr_econ, emr_mtctw_s1, emr_mtctb_s1, emr_mortw_s1, emr_mortb_s1, emr_trtw_s1, emr_trtb_s1, emr_econ_sens, emr_esens_outs)
eur_sens_tab=sens_tables(eur_s1_cent, eur_econ, eur_mtctw_s1, eur_mtctb_s1, eur_mortw_s1, eur_mortb_s1, eur_trtw_s1, eur_trtb_s1, eur_econ_sens, eur_esens_outs)
sear_sens_tab=sens_tables(sear_s1_cent, sear_econ, sear_mtctw_s1, sear_mtctb_s1, sear_mortw_s1, sear_mortb_s1, sear_trtw_s1, sear_trtb_s1, sear_econ_sens, sear_esens_outs)
wpr_sens_tab=sens_tables(wpr_s1_cent, wpr_econ, wpr_mtctw_s1, wpr_mtctb_s1, wpr_mortw_s1, wpr_mortb_s1, wpr_trtw_s1, wpr_trtb_s1, wpr_econ_sens, wpr_esens_outs)
global_sens_tab=sens_tables(global_s1_cent, global_econ, global_mtctw_s1, global_mtctb_s1, global_mortw_s1, global_mortb_s1, global_trtw_s1, global_trtb_s1, global_econ_sens, global_esens_outs)


with pd.ExcelWriter("Table_Sensitivity_fc.xlsx", engine="xlsxwriter") as w:
    afr_sens_tab.to_excel(w, sheet_name="AFR")
    amr_sens_tab.to_excel(w, sheet_name="AMR")
    emr_sens_tab.to_excel(w, sheet_name="EMR")
    eur_sens_tab.to_excel(w, sheet_name="EUR")
    sear_sens_tab.to_excel(w, sheet_name="SEAR")
    wpr_sens_tab.to_excel(w, sheet_name="WPR")
    global_sens_tab.to_excel(w, sheet_name="Global")

print("Sensitivity table saved to Table_Sensitivity_fc.xlsx")


#%% ============================================================
#   乙肝消除目标达成年份分析（Incidence / Mortality）
#   WHO 2030 目标：vs 2015，CHB 发病率 ↓90%，死亡率 ↓65%
# ============================================================

def elimination_year(cent_dict, pop_dict, inc_red=0.90, mort_red=0.65, baseline_year=2015):
    """
    判定消除目标达成年份。
    参数:
        cent_dict : 含 'chb_inc' 和 'mort' 的字典（中央估计，长度110）
        pop_dict  : 含 'pop' 的字典（同一情景的人口，用于算每10万率）
        inc_red   : 发病率相对 baseline_year 的下降阈值（默认 0.90 = 下降 90%）
        mort_red  : 死亡率相对 baseline_year 的下降阈值（默认 0.65 = 下降 65%）
        baseline_year : 基准年（默认 2015）
    返回: (inc_year, mort_year)，若未达成则返回 "未达成"
    """
    year_arr = np.arange(1990, 2100, 1)
    
    # 拿到人口数组：优先 pop_dict["pop"]，若无则尝试 cent_dict["pop"]
    if "pop" in pop_dict:
        pop_arr = pop_dict["pop"]
    elif "pop" in cent_dict:
        pop_arr = cent_dict["pop"]
    else:
        # 两个字典都没 pop（例如 global_*_cent）
        # 退而求其次：用绝对数而非人均率比较降幅
        pop_arr = np.ones(110)
    
    # 每10万人率（pop_arr=1 时退化为绝对数）
    inc_rate = cent_dict["chb_inc"] / pop_arr * 1e5
    mort_rate = cent_dict["mort"] / pop_arr * 1e5
    
    # 基准年的值
    idx_base = baseline_year - 1990   # 2015 → index 25
    inc_base = inc_rate[idx_base]
    mort_base = mort_rate[idx_base]
    
    # 目标值
    inc_target = inc_base * (1 - inc_red)
    mort_target = mort_base * (1 - mort_red)
    
    # 找第一次小于等于目标的年份（只找 2015 之后）
    inc_year = "未达成"
    mort_year = "未达成"
    for i in range(idx_base, len(year_arr)):
        if inc_year == "未达成" and inc_rate[i] <= inc_target:
            inc_year = int(year_arr[i])
        if mort_year == "未达成" and mort_rate[i] <= mort_target:
            mort_year = int(year_arr[i])
    
    return inc_year, mort_year


# ===== 给 global 字典补上 pop 键（让 Global 也能用人均率）=====
for (cent_dict, *regions) in [
    (global_bl_cent, afr_bl_cent, amr_bl_cent, emr_bl_cent, eur_bl_cent, sear_bl_cent, wpr_bl_cent),
    (global_s1_cent, afr_s1_cent, amr_s1_cent, emr_s1_cent, eur_s1_cent, sear_s1_cent, wpr_s1_cent),
    (global_s2_cent, afr_s2_cent, amr_s2_cent, emr_s2_cent, eur_s2_cent, sear_s2_cent, wpr_s2_cent),
    (global_s3_cent, afr_s3_cent, amr_s3_cent, emr_s3_cent, eur_s3_cent, sear_s3_cent, wpr_s3_cent),
]:
    if "pop" not in cent_dict:
        cent_dict["pop"] = sum(r["pop"] for r in regions)

# 对每个区域、每个情景生成消除年份
regions_data = {
    "AFR": (afr_bl_cent, afr_s1_cent, afr_s2_cent, afr_s3_cent),
    "AMR": (amr_bl_cent, amr_s1_cent, amr_s2_cent, amr_s3_cent),
    "EMR": (emr_bl_cent, emr_s1_cent, emr_s2_cent, emr_s3_cent),
    "EUR": (eur_bl_cent, eur_s1_cent, eur_s2_cent, eur_s3_cent),
    "SEAR": (sear_bl_cent, sear_s1_cent, sear_s2_cent, sear_s3_cent),
    "WPR": (wpr_bl_cent, wpr_s1_cent, wpr_s2_cent, wpr_s3_cent),
    "Global": (global_bl_cent, global_s1_cent, global_s2_cent, global_s3_cent),
}

scenarios = ["Baseline", "FC Slow (15y)", "FC Mid (10y)", "FC Fast (5y)"]
elim_rows = []

for reg_name, (bl, s1, s2, s3) in regions_data.items():
    for scen_name, cent in zip(scenarios, (bl, s1, s2, s3)):
        inc_y, mort_y = elimination_year(cent, cent)   # pop 用自身字典
        elim_rows.append({
            "Region": reg_name,
            "Scenario": scen_name,
            "Incidence elimination (-90% vs 2015)": inc_y,
            "Mortality elimination (-65% vs 2015)": mort_y,
        })

elim_df = pd.DataFrame(elim_rows)

# 再生成一个对比表：每情景相对 Baseline 提前了多少年
advance_rows = []
for reg_name in regions_data:
    bl_row = elim_df[(elim_df["Region"]==reg_name) & (elim_df["Scenario"]=="Baseline")].iloc[0]
    for scen_name in ["FC Slow (15y)", "FC Mid (10y)", "FC Fast (5y)"]:
        sc_row = elim_df[(elim_df["Region"]==reg_name) & (elim_df["Scenario"]==scen_name)].iloc[0]
        
        def diff(bl_v, sc_v):
            if bl_v == "未达成" and sc_v == "未达成":
                return "双方均未达成"
            if bl_v == "未达成":
                return f"仅情景达成({sc_v})"
            if sc_v == "未达成":
                return "情景未能达成"
            return bl_v - sc_v   # 正数 = 提前 X 年
        
        advance_rows.append({
            "Region": reg_name,
            "Scenario": scen_name,
            "Incidence 提前年数": diff(bl_row["Incidence elimination (-90% vs 2015)"],
                                        sc_row["Incidence elimination (-90% vs 2015)"]),
            "Mortality 提前年数": diff(bl_row["Mortality elimination (-65% vs 2015)"],
                                        sc_row["Mortality elimination (-65% vs 2015)"]),
        })

advance_df = pd.DataFrame(advance_rows)

with pd.ExcelWriter("Table_Elimination_fc.xlsx", engine="xlsxwriter") as w:
    elim_df.to_excel(w, sheet_name="达成年份", index=False)
    advance_df.to_excel(w, sheet_name="相对基线提前", index=False)

print("Elimination table saved to Table_Elimination_fc.xlsx")
print(elim_df.to_string(index=False))
print()
print("相对基线提前:")
print(advance_df.to_string(index=False))
#%% ============================================================
#   方向 3 升级版：HepB-BD × HepB3 × FC 三参数联合反推
#   疫苗（上游阻断新感染）+ 功能性治愈（下游清理存量）
# ============================================================

print("\n" + "="*70)
print("三参数联合扫描：HepB-BD × HepB3 × FC 治愈率")
print("="*70)

# ---- 网格设置（设计上刻意降低维度，避免组合爆炸）----
# 为了可控，用 "2×2×4 = 16 组合 × 6 区域 = 96 次模拟" 的网格
# 您可以根据需要扩展

bd_levels  = [0.30, 0.90]          # HepB-BD 覆盖（30% 温和 / 90% 激进）
hb3_levels = [0.90]                 # HepB3 只扫一个值（90%，已经接近当前）
fc_eff_levels = [0.10, 0.30, 0.50, 0.70]   # FC 年治愈率（扩到0.7已很高）

# 固定参数
fc_cov_fixed = 0.80    # 固定 FC 覆盖率 80%
rollout = 10
start = 2025
target_years = [2030, 2040, 2050]
inc_reduction = 0.90
mort_reduction = 0.65

pop_fc = ["0-4M","0-4F","5-14M","5-14F","15-49M","15-49F",
          "50-69M","50-69F","70+M","70+F"]
pop_u5 = ["0-4M","0-4F"]   # 疫苗只对新生儿/幼儿组起作用


def run_combo_scenario(F, databook, calib, bd_cov, hb3_cov, fc_eff,
                        fc_cov=0.80, rollout=10, start=2025):
    """联合情景：覆盖 bd + hb3 + fc_cov + fc_eff"""
    P = at.Project(framework=F, databook=databook,
                   sim_start=1990, sim_end=2100, sim_dt=0.25, do_run=False)
    cal = P.make_parset()
    cal.load_calibration(calib)
    
    end = start + rollout
    scen = at.ParameterScenario(name=f"bd{bd_cov}_hb3{hb3_cov}_fc{fc_eff}")
    
    # 1) HepB-BD 覆盖率：2025 起 5 年内升到目标
    for pop in pop_u5:
        scen.add("bd", pop,
                 [1990, start, start+5, 2099],
                 [0.05, 0.15, bd_cov,   bd_cov])   # 起点 0.05 假设，您可以按实际调整
    
    # 2) HepB3 覆盖率：2025 起 5 年内升到目标
    for pop in pop_u5:
        scen.add("hb3", pop,
                 [1990, start, start+5, 2099],
                 [0.50, 0.73, hb3_cov, hb3_cov])
    
    # 3) FC 覆盖率（固定 80%）
    for pop in pop_fc:
        scen.add("fc_cov", pop,
                 [1990, start, end, 2099],
                 [0,    0,     fc_cov, fc_cov])
    
    # 4) FC 治愈率（扫描旋钮）
    for pop in pop_fc:
        scen.add("fc_eff", pop,
                 [1990, 2099],
                 [fc_eff, fc_eff])
    
    res = scen.run(P, parset=cal)
    
    out = [
        {"pop":["alive", "total", "sum"]},
        {"chb_inc":["tot_inc", "total", "sum"]},
        {"mort":[":dd_hbv", "total", "sum"]},
    ]
    result = {}
    for item in out:
        for k, v in item.items():
            arr = at.PlotData(res, outputs=v[0], pops=v[1],
                              pop_aggregation=v[2], t_bins=1).series[0].vals
            result[k] = np.nan_to_num(arr, nan=0.0)
    return result


def check_elim(result_dict, inc_red=0.90, mort_red=0.65, baseline_year=2015):
    year_arr = np.arange(1990, 2100, 1)
    pop_arr = result_dict["pop"]
    inc_rate = result_dict["chb_inc"] / pop_arr * 1e5
    mort_rate = result_dict["mort"] / pop_arr * 1e5
    idx_base = baseline_year - 1990
    inc_target = inc_rate[idx_base] * (1 - inc_red)
    mort_target = mort_rate[idx_base] * (1 - mort_red)
    inc_y, mort_y = None, None
    for i in range(idx_base, len(year_arr)):
        if inc_y is None and inc_rate[i] <= inc_target:
            inc_y = int(year_arr[i])
        if mort_y is None and mort_rate[i] <= mort_target:
            mort_y = int(year_arr[i])
    return inc_y, mort_y


regions_scan = [
    ("AFR",  "databooks/AFR.xlsx",  "calibrations/AFR_calib.xlsx"),
    ("AMR",  "databooks/AMR.xlsx",  "calibrations/AMR_calib.xlsx"),
    ("EMR",  "databooks/EMR.xlsx",  "calibrations/EMR_calib.xlsx"),
    ("EUR",  "databooks/EUR.xlsx",  "calibrations/EUR_calib.xlsx"),
    ("SEAR", "databooks/SEAR.xlsx", "calibrations/SEAR_calib.xlsx"),
    ("WPR",  "databooks/WPR.xlsx",  "calibrations/WPR_calib.xlsx"),
]

scan_rows = []
total = len(regions_scan) * len(bd_levels) * len(hb3_levels) * len(fc_eff_levels)
cnt = 0
for reg_name, db, cal_path in regions_scan:
    for bd_c in bd_levels:
        for hb3_c in hb3_levels:
            for fc_e in fc_eff_levels:
                cnt += 1
                print(f"[{cnt}/{total}] {reg_name} | BD={bd_c*100:.0f}% | HB3={hb3_c*100:.0f}% | FCeff={fc_e*100:.0f}% ...", end=" ", flush=True)
                try:
                    res = run_combo_scenario(F, db, cal_path, bd_c, hb3_c, fc_e,
                                              fc_cov=fc_cov_fixed, rollout=rollout, start=start)
                    inc_y, mort_y = check_elim(res, inc_reduction, mort_reduction)
                    print(f"inc={inc_y}, mort={mort_y}")
                    scan_rows.append({
                        "Region": reg_name,
                        "HepB-BD Coverage": bd_c,
                        "HepB3 Coverage": hb3_c,
                        "FC Annual Cure Rate": fc_e,
                        "Incidence Elim Year": inc_y if inc_y else 9999,
                        "Mortality Elim Year": mort_y if mort_y else 9999,
                    })
                except Exception as e:
                    print(f"FAILED: {e}")

combo_df = pd.DataFrame(scan_rows)
combo_display = combo_df.copy()
for col in ["Incidence Elim Year", "Mortality Elim Year"]:
    combo_display[col] = combo_display[col].apply(
        lambda x: "Not achieved" if x == 9999 else int(x))


# 总结：每区域 × 每目标年，找可行组合
summary_rows = []
for reg_name, _, _ in regions_scan:
    sub = combo_df[combo_df["Region"] == reg_name]
    for target in target_years:
        for ind, col in [("Incidence ↓90%", "Incidence Elim Year"),
                         ("Mortality ↓65%", "Mortality Elim Year")]:
            fs = sub[sub[col] <= target]
            if len(fs) == 0:
                summary_rows.append({
                    "Region": reg_name,
                    "Indicator": ind,
                    "Target Year": target,
                    "Feasible": "No",
                    "Min HepB-BD": "-",
                    "Min FC Cure Rate": "-",
                    "# Combos": 0,
                })
            else:
                # 找资源投入最小的组合（BD 最低 × FC eff 最低）
                fs = fs.copy()
                fs["effort"] = fs["HepB-BD Coverage"] + fs["FC Annual Cure Rate"]
                best = fs.loc[fs["effort"].idxmin()]
                summary_rows.append({
                    "Region": reg_name,
                    "Indicator": ind,
                    "Target Year": target,
                    "Feasible": "Yes",
                    "Min HepB-BD": f"{best['HepB-BD Coverage']*100:.0f}%",
                    "Min FC Cure Rate": f"{best['FC Annual Cure Rate']*100:.0f}%",
                    "# Combos": int(len(fs)),
                })
summary_df = pd.DataFrame(summary_rows)


with pd.ExcelWriter("Table_Combo_Scan.xlsx", engine="xlsxwriter") as w:
    combo_display.to_excel(w, sheet_name="全部扫描结果", index=False)
    summary_df.to_excel(w, sheet_name="达成目标总结", index=False)
    # 每个 region 给一张"BD × FC_eff"的矩阵（固定 HB3）
    for reg_name, _, _ in regions_scan:
        sub = combo_df[combo_df["Region"] == reg_name]
        for hb3_c in hb3_levels:
            s2 = sub[sub["HepB3 Coverage"] == hb3_c]
            for kind, col in [("Inc","Incidence Elim Year"), ("Mort","Mortality Elim Year")]:
                mat = s2.pivot(index="HepB-BD Coverage",
                               columns="FC Annual Cure Rate",
                               values=col)
                sheet_name = f"{reg_name}_{kind}"[:31]
                mat.to_excel(w, sheet_name=sheet_name)

print(f"\n扫描完成！共 {cnt} 次模拟")
print("结果保存到 Table_Combo_Scan.xlsx")
print("\n=== 达成目标总结 ===")
print(summary_df.to_string(index=False))



import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ── 输出目录 ──────────────────────────────────────────────────
_DIR_EPI  = os.path.join(wd, "epi_plots");   os.makedirs(_DIR_EPI,  exist_ok=True)
_DIR_ECON = os.path.join(wd, "econ_plots");  os.makedirs(_DIR_ECON, exist_ok=True)
_DIR_NEW  = os.path.join(wd, "new_figures"); os.makedirs(_DIR_NEW,  exist_ok=True)

# ── Lancet 色板 ────────────────────────────────────────────
_LC = {
    "sq":    "#1A1A2E",   # 深近黑 — Status Quo
    "s1":    "#C0392B",   # 深红   — FC Slow
    "s2":    "#2471A3",   # 深蓝   — FC Mid
    "s3":    "#1A8A5A",   # 深绿   — FC Fast（虚线）
    "calib": "#2C52A0",   # 深蓝   — 校准数据点
}
_BAND = 0.15   # CrI 透明度

import matplotlib as _mpl
import matplotlib.font_manager as _fm
import os as _os

# 清除字体缓存确保能扫描到新安装的字体
_cache_dir = _mpl.get_cachedir()
for _cf in _os.listdir(_cache_dir):
    if "font" in _cf.lower():
        try:
            _os.remove(_os.path.join(_cache_dir, _cf))
        except Exception:
            pass

# 优先 Calibri → 次选 Carlito（与 Calibri 字形完全兼容的开源替代）→ 最终回退
_calibri_paths = [p for p in _fm.findSystemFonts() if "calibri" in p.lower()]
_carlito_paths  = [p for p in _fm.findSystemFonts() if "carlito" in p.lower()]
if _calibri_paths:
    _FONT_FAMILY = _fm.FontProperties(fname=_calibri_paths[0]).get_name()
elif _carlito_paths:
    _FONT_FAMILY = _fm.FontProperties(fname=_carlito_paths[0]).get_name()
else:
    _FONT_FAMILY = "DejaVu Sans"
print(f"[Font] Using: {_FONT_FAMILY}")

plt.rcParams.update({
    "font.family":       _FONT_FAMILY,
    "font.size":         11,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

# ── 通用辅助函数 ───────────────────────────────────────────
_YEAR = np.arange(1990, 2100, 1)

def _clean(arr):
    return np.nan_to_num(np.asarray(arr, dtype=float).flatten(),
                         nan=0.0, posinf=0.0, neginf=0.0)

def _cleana(arr):
    return np.nan_to_num(np.asarray(arr, dtype=float),
                         nan=0.0, posinf=0.0, neginf=0.0)

def _band(ax, psa, col, start=0, end=110, scale=1.0):
    lo = np.percentile(psa[start:end], 2.5,  axis=-1) * scale
    hi = np.percentile(psa[start:end], 97.5, axis=-1) * scale
    ax.fill_between(_YEAR[start:end], lo, hi,
                    color=col, alpha=_BAND, linewidth=0)

def _band_log(ax, psa, col, start, end, scale=1.0):
    lo = np.percentile(psa[start:end], 2.5,  axis=-1) * scale
    hi = np.percentile(psa[start:end], 97.5, axis=-1) * scale
    lo = np.where(lo <= 0, np.nan, lo)
    hi = np.where(hi <= 0, np.nan, hi)
    ax.fill_between(_YEAR[start:end], lo, hi,
                    color=col, alpha=_BAND, linewidth=0)

def _calib_eb(ax, df, col, scale=1.0, unit=1.0, err_frac=0.20):
    """校准数据点 + 误差棒（±err_frac 代表 GBD 不确定性）"""
    if col not in df.columns:
        return
    yrs  = df["year"].values
    vals = df[col].values.astype(float) * scale / unit
    err  = np.abs(vals) * err_frac
    ax.errorbar(yrs, vals, yerr=err, fmt="D", color=_LC["calib"],
                markersize=6, capsize=4, capthick=1.2,
                linewidth=1.2, zorder=6)

def _panel_label(ax, letter):
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", fontfamily=_FONT_FAMILY)

def _neb_year(neb_c, start=2027):
    for y, v in zip(_YEAR, neb_c):
        if y >= start and v > 0:
            return int(y)
    return None

def _gdp_for_reg(reg):
    """从 costs.xlsx 读取区域 GDP/cap，用于 WTP 线"""
    _pr = pd.read_excel("cost and agg data/costs.xlsx", sheet_name="emp_gdp_lex")
    _map = dict(zip(["AFR","AMR","EMR","EUR","SEAR","WPR"],
                    _pr.iloc[1, 1:7].values.astype(float)))
    return _map.get(reg, float(np.mean(list(_map.values()))))


# ════════════════════════════════════════════════════════════
# Figure 1 增强版  epi_plots_v2()
# ════════════════════════════════════════════════════════════
def epi_plots_v2(cbl, cs1, cs2, cs3, bl, s1, s2, s3, wd, calib, reg):
    import matplotlib.ticker as mtick

    plot_prev    = pd.read_excel(wd + calib, sheet_name="pop_prv")
    plot_prev_u5 = pd.read_excel(wd + calib, sheet_name="u5_prev")
    plot_death   = pd.read_excel(wd + calib, sheet_name="t_deaths")
    plot_hcc     = pd.read_excel(wd + calib, sheet_name="hcc_inc")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax_A, ax_B, ax_C, ax_D) = axes.flatten()

    # ── 图例 handles ─────────────────────────────────────
    _handles = [
        mlines.Line2D([], [], color=_LC["sq"],    lw=3.5, ls="-",  label="Status Quo"),
        mlines.Line2D([], [], color=_LC["s1"],    lw=2.0, ls="-",  label="FC Slow (15y)"),
        mlines.Line2D([], [], color=_LC["s2"],    lw=2.0, ls="--", label="FC Mid (10y)"),
        mlines.Line2D([], [], color=_LC["s3"],    lw=2.0, ls=":",  label="FC Fast (5y)"),
        mlines.Line2D([], [], color=_LC["calib"], lw=0,   marker="D", markersize=6,
                      label="Calibration data ±20% (GBD)"),
    ]

    def _plot4(ax, key, cent_bl, cent_s1, cent_s2, cent_s3,
               psa_bl, psa_s1, psa_s2, psa_s3, scale=1.0):
        _band(ax, psa_bl[key] * scale, _LC["sq"])
        _band(ax, psa_s1[key] * scale, _LC["s1"])
        _band(ax, psa_s2[key] * scale, _LC["s2"])
        _band(ax, psa_s3[key] * scale, _LC["s3"])
        ax.plot(_YEAR, cent_bl[key] * scale, color=_LC["sq"], lw=3.5, ls="-",  zorder=5)
        ax.plot(_YEAR, cent_s1[key] * scale, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
        ax.plot(_YEAR, cent_s2[key] * scale, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
        ax.plot(_YEAR, cent_s3[key] * scale, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)

    # A: 总人口流行率
    _plot4(ax_A, "prev", cbl, cs1, cs2, cs3, bl, s1, s2, s3, scale=1e2)
    _calib_eb(ax_A, plot_prev, reg, scale=1e2)
    ax_A.set_ylim(bottom=0)
    ax_A.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_A.set_title("HBsAg seroprevalence (total population)",
                   fontsize=12, fontweight="bold")
    ax_A.set_ylabel("HBsAg seroprevalence (%)", fontsize=11, fontweight="bold")
    ax_A.set_xlim(2000, 2099)

    # B: <5 岁流行率
    _plot4(ax_B, "prev_u5", cbl, cs1, cs2, cs3, bl, s1, s2, s3, scale=1e2)
    _calib_eb(ax_B, plot_prev_u5, reg, scale=1e2)
    ax_B.set_ylim(bottom=0)
    ax_B.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_B.set_title("HBsAg seroprevalence (children <5 years)",
                   fontsize=12, fontweight="bold")
    ax_B.set_ylabel("HBsAg seroprevalence (%)", fontsize=11, fontweight="bold")
    ax_B.set_xlim(2000, 2099)

    # C: HBV 归因死亡
    _plot4(ax_C, "mort", cbl, cs1, cs2, cs3, bl, s1, s2, s3, scale=1/1e3)
    _calib_eb(ax_C, plot_death, reg, unit=1e3)
    ax_C.set_ylim(bottom=0)
    ax_C.set_title("HBV-attributable deaths per year",
                   fontsize=12, fontweight="bold")
    ax_C.set_ylabel("Annual deaths (thousands)", fontsize=11, fontweight="bold")
    ax_C.set_xlim(2000, 2099)

    # D: HCC 发病
    _plot4(ax_D, "hcc_inc", cbl, cs1, cs2, cs3, bl, s1, s2, s3, scale=1/1e3)
    _calib_eb(ax_D, plot_hcc, reg, unit=1e3)
    ax_D.set_ylim(bottom=0)
    ax_D.set_title("HCC incidence per year",
                   fontsize=12, fontweight="bold")
    ax_D.set_ylabel("Annual HCC cases (thousands)", fontsize=11, fontweight="bold")
    ax_D.set_xlim(2000, 2099)

    for ax, lbl in zip(axes.flatten(), ["A","B","C","D"]):
        _panel_label(ax, lbl)
        ax.set_xlabel("Year", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)

    fig.legend(handles=_handles, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    # fig.suptitle removed (Fig1)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = os.path.join(_DIR_EPI, f"{reg}_epi plot PSA_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if reg == "global":
        fig.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig1] {reg} → {out}")

# ════════════════════════════════════════════════════════════
# Coverage plots 增强版  cov_plots_v2()
# ════════════════════════════════════════════════════════════
def cov_plots_v2(cbl, cs1, cs2, cs3, bl, s1, s2, s3, wd, calib, reg):
    """Enhanced coverage plot matching epi_plots_v2 style:
    bold A-D panel subtitles, _LC colour palette, 95% CrI bands.
    Only produced for non-global regions (same logic as original).
    """
    if reg == "global":
        return

    import matplotlib.ticker as mtick

    plot_bd  = pd.read_excel(wd + calib, sheet_name="bd_vax")
    plot_hb3 = pd.read_excel(wd + calib, sheet_name="hb3_vax")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax_A, ax_B, ax_C, ax_D) = axes.flatten()

    _handles = [
        mlines.Line2D([], [], color=_LC["sq"], lw=3.5, ls="--", label="Status Quo"),
        mlines.Line2D([], [], color=_LC["s1"], lw=2.0, ls="-",  label="FC Slow (15y)"),
        mlines.Line2D([], [], color=_LC["s2"], lw=2.0, ls="--", label="FC Mid (10y)"),
        mlines.Line2D([], [], color=_LC["s3"], lw=2.0, ls=":",  label="FC Fast (5y)"),
    ]

    # ── Panel A: HepB Birth-Dose coverage ────────────────
    _band(ax_A, bl["bd_cov"],  _LC["sq"], scale=1e2)
    _band(ax_A, s1["bd_cov"],  _LC["s1"], scale=1e2)
    _band(ax_A, s2["bd_cov"],  _LC["s2"], scale=1e2)
    _band(ax_A, s3["bd_cov"],  _LC["s3"], scale=1e2)
    ax_A.plot(_YEAR, cbl["bd_cov"] * 1e2, color=_LC["sq"], lw=3.5, ls="--", zorder=5)
    ax_A.plot(_YEAR, cs1["bd_cov"] * 1e2, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax_A.plot(_YEAR, cs2["bd_cov"] * 1e2, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax_A.plot(_YEAR, cs3["bd_cov"] * 1e2, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)
    if reg in plot_bd.columns:
        ax_A.scatter(plot_bd["year"], plot_bd[reg] * 1e2,
                     color=_LC["calib"], marker="D", s=30, zorder=6)
    ax_A.axhline(90, color="#C0392B", lw=1.2, ls=":", alpha=0.8)
    ax_A.set_title("Hepatitis B birth-dose vaccine coverage",
                   fontsize=12, fontweight="bold")
    ax_A.set_ylabel("Coverage (%)", fontsize=11, fontweight="bold")

    # ── Panel B: HepB3 (three-dose infant) coverage ──────
    _band(ax_B, bl["hb3_cov"], _LC["sq"], scale=1e2)
    _band(ax_B, s1["hb3_cov"], _LC["s1"], scale=1e2)
    _band(ax_B, s2["hb3_cov"], _LC["s2"], scale=1e2)
    _band(ax_B, s3["hb3_cov"], _LC["s3"], scale=1e2)
    ax_B.plot(_YEAR, cbl["hb3_cov"] * 1e2, color=_LC["sq"], lw=3.5, ls="--", zorder=5)
    ax_B.plot(_YEAR, cs1["hb3_cov"] * 1e2, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax_B.plot(_YEAR, cs2["hb3_cov"] * 1e2, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax_B.plot(_YEAR, cs3["hb3_cov"] * 1e2, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)
    if reg in plot_hb3.columns:
        ax_B.scatter(plot_bd["year"], plot_hb3[reg] * 1e2,
                     color=_LC["calib"], marker="D", s=30, zorder=6)
    ax_B.axhline(90, color="#C0392B", lw=1.2, ls=":", alpha=0.8)
    ax_B.set_title("Infant three-dose HepB vaccine coverage",
                   fontsize=12, fontweight="bold")
    ax_B.set_ylabel("Coverage (%)", fontsize=11, fontweight="bold")

    # ── Panel C: Diagnosis coverage ───────────────────────
    _band(ax_C, bl["dx_prop"],  _LC["sq"], scale=1e2)
    _band(ax_C, s1["dx_prop"],  _LC["s1"], scale=1e2)
    _band(ax_C, s2["dx_prop"],  _LC["s2"], scale=1e2)
    _band(ax_C, s3["dx_prop"],  _LC["s3"], scale=1e2)
    ax_C.plot(_YEAR, cbl["dx_prop"] * 1e2, color=_LC["sq"], lw=3.5, ls="--", zorder=5)
    ax_C.plot(_YEAR, cs1["dx_prop"] * 1e2, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax_C.plot(_YEAR, cs2["dx_prop"] * 1e2, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax_C.plot(_YEAR, cs3["dx_prop"] * 1e2, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)
    ax_C.axhline(90, color="#C0392B", lw=1.2, ls=":", alpha=0.8)
    ax_C.set_title("CHB diagnosis coverage (of all infected)",
                   fontsize=12, fontweight="bold")
    ax_C.set_ylabel("Coverage (%)", fontsize=11, fontweight="bold")

    # ── Panel D: Treatment coverage ───────────────────────
    _band(ax_D, bl["tx_prop"],  _LC["sq"], scale=1e2)
    _band(ax_D, s1["tx_prop"],  _LC["s1"], scale=1e2)
    _band(ax_D, s2["tx_prop"],  _LC["s2"], scale=1e2)
    _band(ax_D, s3["tx_prop"],  _LC["s3"], scale=1e2)
    ax_D.plot(_YEAR, cbl["tx_prop"] * 1e2, color=_LC["sq"], lw=3.5, ls="--", zorder=5)
    ax_D.plot(_YEAR, cs1["tx_prop"] * 1e2, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax_D.plot(_YEAR, cs2["tx_prop"] * 1e2, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax_D.plot(_YEAR, cs3["tx_prop"] * 1e2, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)
    ax_D.axhline(80, color="#C0392B", lw=1.2, ls=":", alpha=0.8)
    ax_D.set_title("CHB treatment coverage (of diagnosed eligible)",
                   fontsize=12, fontweight="bold")
    ax_D.set_ylabel("Coverage (%)", fontsize=11, fontweight="bold")

    # ── Shared axis formatting ─────────────────────────────
    for ax, lbl in zip(axes.flatten(), ["A","B","C","D"]):
        # Use wider x-offset than _panel_label(-0.08) to clear the % tick labels
        ax.text(-0.1, 1.12, lbl, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", fontfamily=_FONT_FAMILY)
        ax.set_xlim(2000, 2099)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel("Year", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)

    fig.legend(handles=_handles, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    _DIR_COV = os.path.join(wd, "cov_plots")
    os.makedirs(_DIR_COV, exist_ok=True)
    out = os.path.join(_DIR_COV, f"{reg}_coverage plot_PSA_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CovV2] {reg} → {out}")

# ════════════════════════════════════════════════════════════
# Figure 2 增强版  econ_plots_v2()
# ════════════════════════════════════════════════════════════
def econ_plots_v2(econ_dict, reg):

    cbl_daly = _clean(econ_dict["cbl_daly"])
    cs1_daly = _clean(econ_dict["cs1_daly"])
    cs2_daly = _clean(econ_dict["cs2_daly"])
    cs3_daly = _clean(econ_dict["cs3_daly"])
    cbl_dirc = _clean(econ_dict["cbl_dirc"])
    cs1_dirc = _clean(econ_dict["cs1_dirc"])
    cs2_dirc = _clean(econ_dict["cs2_dirc"])
    cs3_dirc = _clean(econ_dict["cs3_dirc"])
    cs1_icer = _clean(econ_dict["cs1_icer"])
    cs2_icer = _clean(econ_dict["cs2_icer"])
    cs3_icer = _clean(econ_dict["cs3_icer"])
    cs1_neb  = _clean(econ_dict["cs1_neb"])
    cs2_neb  = _clean(econ_dict["cs2_neb"])
    cs3_neb  = _clean(econ_dict["cs3_neb"])

    bl_daly = _cleana(econ_dict["bl_daly"])
    s1_daly = _cleana(econ_dict["s1_daly"])
    s2_daly = _cleana(econ_dict["s2_daly"])
    s3_daly = _cleana(econ_dict["s3_daly"])
    bl_dirc = _cleana(econ_dict["bl_dirc"])
    s1_dirc = _cleana(econ_dict["s1_dirc"])
    s2_dirc = _cleana(econ_dict["s2_dirc"])
    s3_dirc = _cleana(econ_dict["s3_dirc"])
    s1_icer = _cleana(econ_dict["s1_icer"])
    s2_icer = _cleana(econ_dict["s2_icer"])
    s3_icer = _cleana(econ_dict["s3_icer"])
    s1_neb  = _cleana(econ_dict["s1_neb"])
    s2_neb  = _cleana(econ_dict["s2_neb"])
    s3_neb  = _cleana(econ_dict["s3_neb"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_A, ax_B, ax_C, ax_D = axes.flatten()

    _LEG = [
        mlines.Line2D([], [], color=_LC["sq"], lw=3.5, ls="-",  label="Status Quo"),
        mlines.Line2D([], [], color=_LC["s1"], lw=2.0, ls="-",  label="FC Slow (15y)"),
        mlines.Line2D([], [], color=_LC["s2"], lw=2.0, ls="--", label="FC Mid (10y)"),
        mlines.Line2D([], [], color=_LC["s3"], lw=2.0, ls=":",  label="FC Fast (5y)"),
    ]

    # ── Panel A: DALYs ───────────────────────────────────
    ax = ax_A
    _band(ax, bl_daly/1e6, _LC["sq"]); _band(ax, s1_daly/1e6, _LC["s1"])
    _band(ax, s2_daly/1e6, _LC["s2"]); _band(ax, s3_daly/1e6, _LC["s3"])
    ax.plot(_YEAR, cbl_daly/1e6, color=_LC["sq"], lw=3.5, ls="-",  zorder=5)
    ax.plot(_YEAR, cs1_daly/1e6, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax.plot(_YEAR, cs2_daly/1e6, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax.plot(_YEAR, cs3_daly/1e6, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)
    ax.set_xlim(2026, 2099); ax.set_ylim(bottom=0)
    ax.set_ylabel("DALYs (million)", fontsize=11, fontweight="bold")
    ax_A.set_title("Disability Adjusted Life Years",
                   fontsize=12, fontweight="bold")

    # ── Panel B: 医疗成本（标注尖峰） ───────────────────
    ax = ax_B
    _band(ax, bl_dirc/1e9, _LC["sq"]); _band(ax, s1_dirc/1e9, _LC["s1"])
    _band(ax, s2_dirc/1e9, _LC["s2"]); _band(ax, s3_dirc/1e9, _LC["s3"])
    ax.plot(_YEAR, cbl_dirc/1e9, color=_LC["sq"], lw=3.5, ls="-",  zorder=5)
    ax.plot(_YEAR, cs1_dirc/1e9, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax.plot(_YEAR, cs2_dirc/1e9, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax.plot(_YEAR, cs3_dirc/1e9, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)

    # 标注前期投资冲击区间
    ax.axvspan(2027, 2042, color="#E8F4FD", alpha=0.5, zorder=0)
    _s1_peak = float(np.nanmax(cs1_dirc[37:53] / 1e9))
    ax.annotate(
        "FC scale-up\ninvestment shock\n(2027–2042)",
        xy=(2034, _s1_peak * 0.85),
        xytext=(2050, _s1_peak * 1.15),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3),
        fontsize=8.5, color="#444444", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#AAAAAA", alpha=0.9))
    ax.set_xlim(2026, 2099)
    ax.set_ylabel("Direct healthcare costs (US$ billion)", fontsize=11, fontweight="bold")
    ax_B.set_title("Healthcare Costs",
                   fontsize=12, fontweight="bold")

    # ── Panel C: ICER 对数坐标 + WTP 阈值线 ────────────
    ax = ax_C
    ax.set_yscale("log")

    i_st, i_en = 40, 110   # 2030–2099（早期 ICER 因 DALY diff≈0 失真）
    yy_i = _YEAR[i_st:i_en]

    _band_log(ax, s1_icer/1e3, _LC["s1"], i_st, i_en)
    _band_log(ax, s2_icer/1e3, _LC["s2"], i_st, i_en)
    _band_log(ax, s3_icer/1e3, _LC["s3"], i_st, i_en)

    for arr, col, ls in [(cs1_icer, _LC["s1"], "-"),
                          (cs2_icer, _LC["s2"], "--"),
                          (cs3_icer, _LC["s3"], ":")]:
        y = np.where(arr[i_st:i_en] <= 0, np.nan, arr[i_st:i_en] / 1e3)
        ax.plot(yy_i, y, color=col, lw=2.0, ls=ls, zorder=4)

    # WTP 阈值（真实 GDP/cap 来自 costs.xlsx）
    _gdp = _gdp_for_reg(reg)
    ax.axhline(_gdp / 1e3, color="#27AE60", lw=1.4, ls="--", zorder=3,
               label=f"WTP 1× GDP/cap (~${_gdp:,.0f})")
    ax.axhline(_gdp * 0.5 / 1e3, color="#F39C12", lw=1.4, ls=":", zorder=3,
               label=f"WTP 0.5× GDP/cap (~${_gdp*0.5:,.0f})")
    ax.axhspan(0.01, _gdp / 1e3, color="#27AE60", alpha=0.04, zorder=0)

    ax.set_xlim(2030, 2099)
    ax.set_ylabel("Cumulative ICER (US$'000 / DALY averted)", fontsize=11, fontweight="bold")
    ax_C.set_title("ICER",
                   fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _:
            f"${x:.0f}k" if x >= 1 else f"${x*1000:.0f}"))
    _icer_leg = list(_LEG[1:]) + [
        mlines.Line2D([], [], color="#27AE60", lw=1.4, ls="--",
                      label=f"WTP 1× GDP/cap (~${_gdp:,.0f})"),
        mlines.Line2D([], [], color="#F39C12", lw=1.4, ls=":",
                      label=f"WTP 0.5× GDP/cap (~${_gdp*0.5:,.0f})"),
    ]
    ax.legend(handles=_icer_leg, fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(which="both", ls=":", lw=0.4, alpha=0.5)

    # ── Panel D: NEB 加粗零线 + 转正年份标注 ────────────
    ax = ax_D
    n_st = 36   # 2026
    yy_n = _YEAR[n_st:110]

    _band(ax, s1_neb/1e9, _LC["s1"], n_st, 110)
    _band(ax, s2_neb/1e9, _LC["s2"], n_st, 110)
    _band(ax, s3_neb/1e9, _LC["s3"], n_st, 110)

    ax.plot(yy_n, cs1_neb[n_st:110]/1e9, color=_LC["s1"], lw=2.0, ls="-",  zorder=4)
    ax.plot(yy_n, cs2_neb[n_st:110]/1e9, color=_LC["s2"], lw=2.0, ls="--", zorder=4)
    ax.plot(yy_n, cs3_neb[n_st:110]/1e9, color=_LC["s3"], lw=2.0, ls=":",  zorder=4)

    # 加粗零基准线
    ax.axhline(0, color="black", lw=2.5, ls="-", zorder=5,
               label="Break-even (NEB = 0)")

    # 各情景转正年份：动态检测 + 标注
    _neb_data = [
        (cs1_neb, _LC["s1"], "Slow",  1.5, 0.22),
        (cs2_neb, _LC["s2"], "Mid",  -1.5, 0.32),
        (cs3_neb, _LC["s3"], "Fast",  0.0, 0.42),
    ]
    _y_range = (np.nanmax(cs3_neb[n_st:110]/1e9) -
                np.nanmin(cs1_neb[n_st:110]/1e9))
    if _y_range == 0 or np.isnan(_y_range):
        _y_range = 50.0

    for neb_c, col, suffix, x_off, y_frac in _neb_data:
        yr = _neb_year(neb_c)
        if yr is None:
            continue
        ax.scatter([yr], [0], s=75, color=col, zorder=7,
                   edgecolors="white", linewidths=1.5)
        ax.axvline(yr, color=col, lw=1.0, ls=":", alpha=0.8, zorder=3)
        ax.annotate(
            f"{yr}\n({suffix})",
            xy=(yr, 0),
            xytext=(yr + x_off, _y_range * y_frac),
            fontsize=8.5, fontweight="bold", color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.9),
            ha="center")

    ax.set_xlim(2026, 2099)
    ax.set_ylabel("Cumulative net economic benefit (US$ billion)", fontsize=11, fontweight="bold")
    ax_D.set_title("Net Economic Benefit",
                   fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)

    # ── 通用格式 ─────────────────────────────────────────
    for ax_i, lbl in zip(axes.flatten(), ["A","B","C","D"]):
        _panel_label(ax_i, lbl)
        ax_i.set_xlabel("Year", fontsize=11)
        ax_i.tick_params(labelsize=10)

    fig.legend(handles=_LEG, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    # fig.suptitle removed (Fig2)
    plt.tight_layout(rect=[0, 0.04, 1, 1], pad=2.0)

    out = os.path.join(_DIR_ECON, f"{reg}_econ plot PSA_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if reg == "global":
        fig.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig2] {reg} → {out}")


# ════════════════════════════════════════════════════════════
# Figure 3：龙卷风图（基于 global_sens_tab 真实敏感性数据）
# ════════════════════════════════════════════════════════════
def _fig_tornado():
    tab = global_sens_tab.copy().reset_index(drop=True)
    base_icer = float(tab.iloc[0, 7])

    pairs = []
    labels_raw = list(tab.iloc[1:, 0])
    icer_raw   = list(tab.iloc[1:, 7].astype(float))

    # ── 修复：先判断每行是否属于 worst/best 或 Lower/Upper 对 ──────────
    _pair_kws = ("worst", "best", "Lower bound", "Upper bound")
    def _is_paired(label):
        return any(k in str(label) for k in _pair_kws)

    i = 0
    while i < len(labels_raw):
        lab = str(labels_raw[i])

        if _is_paired(lab) and i + 1 < len(labels_raw) and _is_paired(str(labels_raw[i + 1])):
            # 标准双行：取公共标签，lo/hi 取 min/max
            common = lab.split(" worst")[0].split(" best")[0].strip()
            common = common.replace("Lower bound ", "").replace("Upper bound ", "").strip()
            lo = min(icer_raw[i], icer_raw[i + 1])
            hi = max(icer_raw[i], icer_raw[i + 1])
            if abs(hi - lo) >= 1.0:           # 过滤零范围参数
                pairs.append((common, lo, hi))
            i += 2                             # 跳过已消费的两行

        else:
            # 单行条目（如 HBIG + maternal antivirals）
            lbl = lab.replace("Lower bound ", "").replace("Upper bound ", "").strip()
            val = icer_raw[i]
            # 单行只显示与 base 的偏差（若与 base 几乎相同则过滤）
            if abs(val - base_icer) >= 1.0:
                pairs.append((lbl, min(val, base_icer), max(val, base_icer)))
            i += 1                             # 只跳 1 行
    # ──────────────────────────────────────────────────────────────────────

    # 手动添加 FC 药价敏感性（US$2,000 / US$10,000），此行不在 global_sens_tab 中
    # 数据来自论文 Table S7.1 及正文 Section 3.6
    pairs.append(("FC drug price (US$2,000–10,000)", 6_150, 25_200))

    pairs.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
    n = len(pairs)

    # 图宽加大确保 Y 轴长标签 + 右侧 $25,200 均有空间
    fig, ax = plt.subplots(figsize=(13, max(7, n * 0.52 + 2.2)))
    col_lo, col_hi = "#5DADE2", "#E74C3C"

    # 计算全局 x 范围，用于固定偏移量（像素无关）
    all_lo = min(p[1] for p in pairs)
    all_hi = max(p[2] for p in pairs)
    x_range = all_hi - all_lo
    # 固定标签偏移：x 轴总范围的 0.8%，避免大 span 时标签跑出图外
    offset = x_range * 0.008

    for i, (lab, lo, hi) in enumerate(pairs):
        y = n - 1 - i
        ax.barh(y, lo - base_icer, left=base_icer, height=0.62,
                color=col_lo, zorder=3)
        ax.barh(y, hi - base_icer, left=base_icer, height=0.62,
                color=col_hi, zorder=3)

        # lo 标签：若 lo 过小（bar 向左延伸超过 x_range 的 30%），
        # 改为在 bar 内右端显示（白字），避免与 Y 轴标签重叠
        lo_far = (base_icer - lo) > x_range * 0.30
        if lo_far:
            ax.text(lo + offset, y, f"  ${lo:,.0f}",
                    ha="left", va="center", fontsize=7.5,
                    color="white", fontweight="bold", fontfamily=_FONT_FAMILY, zorder=5)
        else:
            ax.text(lo - offset, y, f"${lo:,.0f}",
                    ha="right", va="center", fontsize=7.5,
                    color="#2471A3", fontfamily=_FONT_FAMILY)

        ax.text(hi + offset, y, f"${hi:,.0f}",
                ha="left", va="center", fontsize=7.5,
                color="#C0392B", fontfamily=_FONT_FAMILY)

    ax.set_yticks(range(n))
    ax.set_yticklabels([p[0] for p in pairs[::-1]], fontsize=9, fontweight="bold")
    ax.axvline(base_icer, color="black", lw=1.6, zorder=4)
    ax.text(base_icer, n - 0.12, f"Base\n${base_icer:,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", fontfamily=_FONT_FAMILY)

    # WTP 线（真实 GDP 平均值）
    _gdp = _gdp_for_reg("global")
    ax.axvline(_gdp, color="#27AE60", lw=1.2, ls="--", alpha=0.85, zorder=2)
    ax.text(_gdp + offset * 0.5, -0.75, f"1× GDP/cap\n(~${_gdp:,.0f})",
            color="#27AE60", fontsize=7.5, ha="left", fontfamily=_FONT_FAMILY)

    ax.legend(handles=[
        mpatches.Patch(color=col_lo, label="Favourable assumption (lower ICER)"),
        mpatches.Patch(color=col_hi, label="Unfavourable assumption (higher ICER)"),
    ], fontsize=8.5, loc="lower right", framealpha=0.9)

    ax.set_xlabel("ICER (US\$/DALY averted, 2025 US\$)", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # 给 Y 轴标签留出足够左边距，右侧为 $25,200 标签留空间
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.22, right=0.95)
    out = os.path.join(_DIR_NEW, "Fig3_Tornado.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig3] Tornado → {out}")


# ════════════════════════════════════════════════════════════
# Figure 4：AFR 诊断-治疗级联图
# ════════════════════════════════════════════════════════════
def _fig_cascade():
    yr = 37   # index → 2027

    total   = float(afr_bl_cent["chb_pop"][yr])
    dx_prop = float(afr_bl_cent["dx_prop"][yr])
    tx_prop = float(afr_bl_cent["tx_prop"][yr])

    diagnosed = total * dx_prop
    # FIX: tx_prop = treat_cov 是占全体 CHB 的比例，不是占 diagnosed 的比例
    # 原代码 on_tx = diagnosed * tx_prop 相当于双重缩小，导致结果近乎为零
    on_tx     = total * tx_prop
    # 治疗覆盖率（占已诊断患者中的比例），用于标注
    tx_among_dx = (tx_prop / dx_prop) if dx_prop > 1e-9 else 0.0
    fc_elig   = on_tx * 0.80
    # FC 覆盖率：取 FC Slow 5 年后（2032，index=42）
    fc_cov_val = float(np.clip(afr_s1_cent["fc_cov_val"][min(42, len(afr_s1_cent["fc_cov_val"])-1)], 0, 1))
    fc_treat  = fc_elig * fc_cov_val
    fc_cured  = fc_treat * 0.09   # base-case fc_eff

    nodes   = [total, diagnosed, on_tx, fc_elig, fc_treat, fc_cured]
    labels  = [
        f"CHB Chronic\nInfection (AFR)\n{total/1e6:.1f}M",
        f"Diagnosed\n{diagnosed/1e6:.1f}M\n({100*dx_prop:.0f}% of CHB)",
        f"On Treatment\n{on_tx/1e6:.2f}M\n({100*on_tx/total:.1f}%)",
        f"FC Eligible\n{fc_elig/1e6:.2f}M\n({100*fc_elig/total:.1f}%)",
        f"Receiving FC\n{fc_treat/1e6:.3f}M\n({100*fc_treat/total:.2f}%)",
        f"Functionally\nCured\n{fc_cured/1e3:.0f}k\n({100*fc_cured/total:.3f}%)",
    ]
    colors = ["#2C3E50","#2471A3","#1A5276","#117A65","#1E8449","#F39C12"]

    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.48, 1.1); ax.axis("off")

    xs  = np.linspace(0, 5, 6)
    max_h = 0.70
    bh  = [v / total * max_h for v in nodes]
    bw  = 0.35

    for i in range(5):
        x1, x2 = xs[i], xs[i+1]
        h1, h2 = bh[i], bh[i+1]
        t, b   = 0.5 + h1/2, 0.5 - h1/2
        ax.add_patch(plt.Polygon(
            [(x1+bw/2, b), (x1+bw/2, t),
             (x2-bw/2, 0.5+h2/2), (x2-bw/2, 0.5-h2/2)],
            closed=True, facecolor=colors[i], alpha=0.18,
            edgecolor="none", zorder=1))
        lost = h1 - h2
        if lost > 0.001:
            ax.add_patch(plt.Polygon(
                [(x1+bw/2, b), (x1+bw/2, b+lost),
                 (x2-bw/2, 0.5-h2/2+0.002), (x2-bw/2, 0.5-h2/2)],
                closed=True, facecolor="#BDC3C7", alpha=0.28,
                edgecolor="none", zorder=1))

    for i, (x, h, col, lab) in enumerate(zip(xs, bh, colors, labels)):
        ax.add_patch(plt.Rectangle((x-bw/2, 0.5-h/2), bw, h,
                                   facecolor=col, edgecolor="white",
                                   linewidth=1.2, zorder=3))
        ax.text(x, 0.5+h/2+0.03, lab,
                ha="center", va="bottom", fontsize=7.5,
                color="#2C2C2C", fontweight="bold", zorder=5, fontfamily=_FONT_FAMILY)

    # 标注两个主要瓶颈
    ax.annotate(f"Diagnosis gap\n({100*(1-dx_prop):.0f}% undiagnosed)",
                xy=(xs[1]-bw/2, 0.5), xytext=(xs[1]-0.72, 0.10),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
                fontsize=8, color="#E74C3C", fontweight="bold", ha="center")
    ax.annotate(f"Treatment gap\n({100*(1-tx_among_dx):.0f}% of dx\nnot treated)",
                xy=(xs[2]-bw/2, 0.5), xytext=(xs[2]-0.72, 0.00),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
                fontsize=8, color="#E74C3C", fontweight="bold", ha="center")

    for x, sl in zip(xs, ["Total\nCHB", "Diagnosed", "On\nTreatment",
                           "FC\nEligible", "Receiving\nFC", "Functionally\nCured"]):
        ax.text(x, -0.14, sl, ha="center", va="top", fontsize=8,
                color="#555555", style="italic", fontfamily=_FONT_FAMILY, fontweight="bold")

    # ax.set_title removed (Fig4)

    fig.tight_layout(pad=1.5)
    out = os.path.join(_DIR_NEW, "Fig4_Cascade.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig4] Cascade → {out}")


# ════════════════════════════════════════════════════════════
# Figure 5：BD × FC_eff 热力图（来自 combo_df 真实扫描）
# ════════════════════════════════════════════════════════════
def _fig_elim_matrix():
    """
    Figure 5 — WHO Elimination Target Achievement Matrix

    For each WHO region × scenario, shows the year when:
      - CHB incidence falls ≥90% vs 2015  (Panel A)
      - HBV-attributable mortality falls ≥65% vs 2015  (Panel B)

    Uses existing central-estimate outputs (no new simulation required).
    Core message: Without FC, several high-burden regions cannot achieve WHO
    elimination targets within this century; FC added to a strengthened care
    cascade advances the target year and in some regions is the decisive factor.
    """
    BASELINE_YEAR = 2015
    idx_base = BASELINE_YEAR - 1990
    NOT_ACHIEVED = 2100          # sentinel for plotting "not achieved by 2099"
    XLIM = (2025, 2101)

    regions_order = ["AFR", "SEAR", "WPR", "EMR", "AMR", "EUR"]
    cent_map = {
        "AFR":  {"bl": afr_bl_cent,  "s1": afr_s1_cent,
                 "s2": afr_s2_cent,  "s3": afr_s3_cent},
        "AMR":  {"bl": amr_bl_cent,  "s1": amr_s1_cent,
                 "s2": amr_s2_cent,  "s3": amr_s3_cent},
        "EMR":  {"bl": emr_bl_cent,  "s1": emr_s1_cent,
                 "s2": emr_s2_cent,  "s3": emr_s3_cent},
        "EUR":  {"bl": eur_bl_cent,  "s1": eur_s1_cent,
                 "s2": eur_s2_cent,  "s3": eur_s3_cent},
        "SEAR": {"bl": sear_bl_cent, "s1": sear_s1_cent,
                 "s2": sear_s2_cent, "s3": sear_s3_cent},
        "WPR":  {"bl": wpr_bl_cent,  "s1": wpr_s1_cent,
                 "s2": wpr_s2_cent,  "s3": wpr_s3_cent},
    }

    scenarios = [
        ("bl", "Status Quo",    _LC["sq"],  "o",  8.0, "--"),
        ("s1", "FC Slow (15y)", _LC["s1"],  "D",  7.5, "-"),
        ("s2", "FC Mid (10y)",  _LC["s2"],  "s",  7.5, "-"),
        ("s3", "FC Fast (5y)",  _LC["s3"],  "^",  8.0, "-"),
    ]

    def _find_elim_year(cent, key, target_frac, idx_base):
        """Return first year where cent[key]/cent[pop] drops by target_frac vs 2015."""
        rate = cent[key] / cent["pop"]
        thresh = rate[idx_base] * (1.0 - target_frac)
        for i, yr in enumerate(_YEAR):
            if yr >= BASELINE_YEAR and rate[i] <= thresh:
                return int(yr)
        return NOT_ACHIEVED

    # ── Compute elimination years ────────────────────────────────────────
    results = {met: {reg: {} for reg in regions_order}
               for met in ("inc", "mort")}

    for reg in regions_order:
        for sc_key, *_ in scenarios:
            cent = cent_map[reg][sc_key]
            results["inc"][reg][sc_key]  = _find_elim_year(cent, "chb_inc", 0.90, idx_base)
            results["mort"][reg][sc_key] = _find_elim_year(cent, "mort",    0.65, idx_base)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")

    panel_cfg = [
        (axes[0], "inc",  "CHB incidence elimination  (−90% vs 2015)", 0.90),
        (axes[1], "mort", "HBV mortality elimination  (−65% vs 2015)", 0.65),
    ]

    y_pos   = {reg: i for i, reg in enumerate(regions_order)}
    n_sc    = len(scenarios)
    offsets = np.linspace(-0.28, 0.28, n_sc)   # vertical jitter per scenario

    for ax, met, title, _ in panel_cfg:
        for si, (sc_key, sc_label, col, mkr, ms, ls) in enumerate(scenarios):
            xs, ys, xs_na, ys_na = [], [], [], []
            for reg in regions_order:
                yr = results[met][reg][sc_key]
                yp = y_pos[reg] + offsets[si]
                if yr == NOT_ACHIEVED:
                    xs_na.append(XLIM[1] - 0.5)
                    ys_na.append(yp)
                else:
                    xs.append(yr)
                    ys.append(yp)

            # Achieved dots + connecting lines between scenarios
            ax.scatter(xs, ys, color=col, marker=mkr, s=ms**2,
                       zorder=5, label=sc_label if ax is axes[0] else "")
            # "Not achieved" shown as hollow rightward-pointing triangle at border
            ax.scatter(xs_na, ys_na, color=col, marker=">", s=7**2,
                       edgecolors=col, facecolors="none", lw=1.8,
                       zorder=5)

        # Draw thin lines connecting Status Quo → FC Fast for each region
        for reg in regions_order:
            yp_base = y_pos[reg]
            x_vals = [results[met][reg][sc_key]
                      for sc_key, *_ in scenarios]
            # Replace NOT_ACHIEVED with XLIM[1]-0.5 for line drawing
            x_draw = [x if x < NOT_ACHIEVED else XLIM[1]-0.5 for x in x_vals]
            ax.plot(x_draw,
                    [yp_base + offsets[si] for si in range(n_sc)],
                    color="#BBBBBB", lw=0.8, zorder=2)

        # WHO target reference lines
        ax.axvline(2030, color="#E74C3C", lw=1.2, ls=":", alpha=0.8, zorder=3)
        ax.axvline(2050, color="#F39C12", lw=1.2, ls=":", alpha=0.8, zorder=3)
        ax.text(2030, -0.65, "2030\ntarget", color="#E74C3C",
                fontsize=7.5, ha="center", va="top")
        ax.text(2050, -0.65, "2050\ntarget", color="#F39C12",
                fontsize=7.5, ha="center", va="top")

        # "Not achieved" annotation
        ax.axvline(XLIM[1]-0.5, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
        ax.text(XLIM[1]-0.3, len(regions_order)-0.5,
                "Not\nachieved\nby 2099",
                fontsize=7.5, color="#888888", ha="left", va="top")

        ax.set_xlim(XLIM)
        ax.set_ylim(-0.9, len(regions_order) - 0.2)
        ax.set_yticks(range(len(regions_order)))
        ax.set_yticklabels(regions_order, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year of achieving WHO elimination target",
                      fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
        ax.invert_yaxis()   # highest-burden region (AFR) at top

    # Shared legend
    handles = [
        mlines.Line2D([], [], color=col, marker=mkr, ms=ms,
                      ls="none", label=sc_label)
        for sc_key, sc_label, col, mkr, ms, _ in scenarios
    ] + [
        mlines.Line2D([], [], color="#BBBBBB", marker=">", ms=7,
                      ls="none", mfc="none", label="Not achieved by 2099"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(_DIR_NEW, "Fig5_ElimMatrix.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if True:
        fig.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig5] ElimMatrix → {out}")


def _fig_heatmap():
    try:
        df = combo_df.copy()
    except NameError:
        print("  [Fig5] 跳过：combo_df 未定义（方向3扫描未运行）")
        return

    regions  = ["AFR","AMR","EMR","EUR","SEAR","WPR"]
    bd_vals  = sorted(df["HepB-BD Coverage"].unique())
    fc_vals  = sorted(df["FC Annual Cure Rate"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.0))
    axes_flat = axes.flatten()

    # Switch to Mortality Elim Year: FC cure rate directly prevents HBV-attributable
    # deaths (liver disease progression). Incidence is driven mainly by vaccination
    # coverage (denominator trap), so FC rate variation would not show in incidence.
    outcome_col = "Mortality Elim Year"
    cmap = plt.cm.RdYlGn_r
    norm = matplotlib.colors.Normalize(vmin=2030, vmax=2100)

    # Axis label layout:
    #   Top row:    AFR(0) ylabel only | AMR(1) xlabel only | EMR(2) no labels
    #   Bottom row: EUR(3) ylabel only | SEAR(4) xlabel only | WPR(5) no labels
    ylabel_idx = {0, 3}
    xlabel_idx = {1, 4}

    for idx, (ax, reg) in enumerate(zip(axes_flat, regions)):
        sub  = df[df["Region"] == reg]
        data = np.full((len(fc_vals), len(bd_vals)), np.nan)
        for ri, fc in enumerate(fc_vals):
            for ci, bd in enumerate(bd_vals):
                row = sub[(sub["FC Annual Cure Rate"] == fc) &
                          (sub["HepB-BD Coverage"]   == bd)]
                if len(row):
                    v = float(row[outcome_col].values[0])
                    data[ri, ci] = v if v < 9999 else np.nan

        masked = np.ma.masked_invalid(data)
        ax.imshow(masked, cmap=cmap, norm=norm,
                  aspect="auto", interpolation="nearest")

        for ri in range(len(fc_vals)):
            for ci in range(len(bd_vals)):
                v = data[ri, ci]
                if np.isnan(v):
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=11, color="#888888", fontfamily=_FONT_FAMILY)
                else:
                    norm_v = (v - 2030) / 70.0
                    tc = "white" if norm_v < 0.25 or norm_v > 0.80 else "black"
                    ax.text(ci, ri, f"{int(v)}", ha="center", va="center",
                            fontsize=10.5, fontweight="bold", color=tc,
                            fontfamily=_FONT_FAMILY)

        ax.set_xticks(range(len(bd_vals)))
        ax.set_xticklabels([f"{int(v*100)}%" for v in bd_vals], fontsize=9)
        ax.set_yticks(range(len(fc_vals)))
        ax.set_yticklabels([f"FC {int(v*100)}%" for v in fc_vals], fontsize=9)
        title_col = "#C0392B" if reg == "EUR" else "#2C2C2C"
        ax.set_title(f"{reg}{'  †' if reg=='EUR' else ''}",
                     fontsize=12, fontweight="bold", color=title_col, pad=4)
        if idx in ylabel_idx:
            ax.set_ylabel("FC annual cure rate", fontsize=10, fontweight="bold")
        if idx in xlabel_idx:
            ax.set_xlabel("HepB birth-dose coverage", fontsize=10, fontweight="bold")

    cb_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cb_ax).set_label(
        "Year of mortality\nelimination (−65% vs 2015)", fontsize=9, labelpad=8, fontweight="bold")

    # fig.suptitle removed (Fig5)
    fig.tight_layout(rect=[0, 0, 0.91, 1])

    out = os.path.join(_DIR_NEW, "Fig5_Heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig5] Heatmap → {out}")


# ════════════════════════════════════════════════════════════
# 执行所有图表
# ════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("开始生成所有专家建议图表（Figure 1–5，含视觉优化版）")
print("="*65)

_cal = "cost and agg data/pop_calib.xlsx"

print("\n─ Figure 1 ─")
for _reg, _bl, _s1, _s2, _s3, _blr, _s1r, _s2r, _s3r in [
    ("AFR",  afr_bl_cent,  afr_s1_cent,  afr_s2_cent,  afr_s3_cent,
             afr_bl_runs,  afr_s1_runs,  afr_s2_runs,  afr_s3_runs),
    ("AMR",  amr_bl_cent,  amr_s1_cent,  amr_s2_cent,  amr_s3_cent,
             amr_bl_runs,  amr_s1_runs,  amr_s2_runs,  amr_s3_runs),
    ("EMR",  emr_bl_cent,  emr_s1_cent,  emr_s2_cent,  emr_s3_cent,
             emr_bl_runs,  emr_s1_runs,  emr_s2_runs,  emr_s3_runs),
    ("EUR",  eur_bl_cent,  eur_s1_cent,  eur_s2_cent,  eur_s3_cent,
             eur_bl_runs,  eur_s1_runs,  eur_s2_runs,  eur_s3_runs),
    ("SEAR", sear_bl_cent, sear_s1_cent, sear_s2_cent, sear_s3_cent,
             sear_bl_runs, sear_s1_runs, sear_s2_runs, sear_s3_runs),
    ("WPR",  wpr_bl_cent,  wpr_s1_cent,  wpr_s2_cent,  wpr_s3_cent,
             wpr_bl_runs,  wpr_s1_runs,  wpr_s2_runs,  wpr_s3_runs),
    ("global", global_bl_cent, global_s1_cent, global_s2_cent, global_s3_cent,
               global_bl_runs, global_s1_runs, global_s2_runs, global_s3_runs),
]:
    epi_plots_v2(_bl, _s1, _s2, _s3, _blr, _s1r, _s2r, _s3r, wd, _cal, _reg)
    cov_plots_v2(_bl, _s1, _s2, _s3, _blr, _s1r, _s2r, _s3r, wd, _cal, _reg)

print("\n─ Figure 2 ─")
for _reg, _econ in [
    ("AFR",  afr_econ),  ("AMR",  amr_econ),  ("EMR",  emr_econ),
    ("EUR",  eur_econ),  ("SEAR", sear_econ), ("WPR",  wpr_econ),
    ("global", global_econ),
]:
    econ_plots_v2(_econ, _reg)

print("\n─ Figure 3 ─")
_fig_tornado()

print("\n─ Figure 4 ─")
_fig_cascade()

print("\n─ Figure 5 ─")
_fig_elim_matrix()
_fig_heatmap()

print("\n" + "="*65)
print("全部完成！输出路径：")
print(f"  Figure 1 (v2)  → {_DIR_EPI}")
print(f"  Figure 2 (v2)  → {_DIR_ECON}")
print(f"  Figure 3–5     → {_DIR_NEW}")
print("="*65)

# ════════════════════════════════════════════════════════════
# Threshold Pricing Analysis
# ════════════════════════════════════════════════════════════
#
# Mathematical derivation
# -----------------------
# FC drug cost stored at base price P0 = $5,000:
#   FC_cost(P0) = sum(cs_fcc[36:110, 0])        [discounted, cumulative]
#
# The per-unit FC weight (person-years × discount factor):
#   N_fc = FC_cost(P0) / P0
#
# Because FC drug cost scales linearly with price P, the ICER at any P is:
#   ICER(P) = [ΔDirect_non_FC  +  N_fc × P]  /  ΔDALYs
#
# where:
#   ΔDirect_non_FC = [sum(cs_dirc) − sum(cbl_dirc)] − FC_cost(P0)
#   ΔDALYs         = sum(cbl_daly) − sum(cs_daly)
#
# Setting ICER(P_thresh) = WTP and solving:
#   P_thresh = (WTP × ΔDALYs − ΔDirect_non_FC) / N_fc
#
# No additional model runs are required — all terms are already in econ_dict.
# ────────────────────────────────────────────────────────────

# Step 1: Add fcc keys to global_econ (they are not yet aggregated there)
global_econ["cbl_fcc"] = (afr_econ["cbl_fcc"] + amr_econ["cbl_fcc"] +
                          emr_econ["cbl_fcc"] + eur_econ["cbl_fcc"] +
                          sear_econ["cbl_fcc"] + wpr_econ["cbl_fcc"])
global_econ["cs1_fcc"] = (afr_econ["cs1_fcc"] + amr_econ["cs1_fcc"] +
                          emr_econ["cs1_fcc"] + eur_econ["cs1_fcc"] +
                          sear_econ["cs1_fcc"] + wpr_econ["cs1_fcc"])
global_econ["cs2_fcc"] = (afr_econ["cs2_fcc"] + amr_econ["cs2_fcc"] +
                          emr_econ["cs2_fcc"] + eur_econ["cs2_fcc"] +
                          sear_econ["cs2_fcc"] + wpr_econ["cs2_fcc"])
global_econ["cs3_fcc"] = (afr_econ["cs3_fcc"] + amr_econ["cs3_fcc"] +
                          emr_econ["cs3_fcc"] + eur_econ["cs3_fcc"] +
                          sear_econ["cs3_fcc"] + wpr_econ["cs3_fcc"])


# Step 2: Define threshold pricing function
def price_threshold_analysis(region_econs, cost_path, fc_base_price=5000,
                             start_idx=36, end_idx=110):
    """
    Compute the maximum FC drug price at which ICER ≤ WTP threshold
    (1× and 0.5× GDP per capita) for each WHO region and scenario.

    Parameters
    ----------
    region_econs  : dict  {region_label: econ_dict}
                    e.g. {"AFR": afr_econ, ..., "Global": global_econ}
    cost_path     : str   path to costs.xlsx (containing 'emp_gdp_lex' sheet)
    fc_base_price : float base-case FC drug price used in the model (default $5,000)
    start_idx     : int   first time-step index for 2026 (default 36, i.e. 1990+36)
    end_idx       : int   one-past-last index for 2099 (default 110)

    Returns
    -------
    df : pd.DataFrame  with columns:
         Region | Scenario | GDP/cap | ICER@$5k | P_thresh_1x | P_thresh_05x
    """
    # Load GDP per capita for each region
    gdp_tab = pd.read_excel(cost_path, sheet_name="emp_gdp_lex", header=None)
    region_labels = list(gdp_tab.iloc[0, 1:7])   # AFR AMR EMR EUR SEAR WPR
    gdp_vals      = list(gdp_tab.iloc[2, 1:7].astype(float))
    gdp_map       = dict(zip(region_labels, gdp_vals))
    # Global: population-weighted average (use simple mean as approximation)
    gdp_map["Global"] = float(np.mean(gdp_vals))

    SL  = slice(start_idx, end_idx)
    P0  = fc_base_price
    rows = []

    scenario_keys = {
        "FC Slow (15y)": ("cs1_fcc", "cbl_dirc", "cs1_dirc",
                          "cbl_daly", "cs1_daly"),
        "FC Mid (10y)":  ("cs2_fcc", "cbl_dirc", "cs2_dirc",
                          "cbl_daly", "cs2_daly"),
        "FC Fast (5y)":  ("cs3_fcc", "cbl_dirc", "cs3_dirc",
                          "cbl_daly", "cs3_daly"),
    }

    for reg, econ in region_econs.items():
        gdp = gdp_map.get(reg, np.nan)
        wtp_1x  = gdp
        wtp_05x = gdp * 0.5

        for scen_label, (fcc_key, bl_dirc_key, cs_dirc_key,
                          bl_daly_key, cs_daly_key) in scenario_keys.items():

            # ── Cumulative FC drug cost at base price P0 ──────────────
            fcc = econ[fcc_key]
            # fcc may be shape (110,1) or (110,) depending on PSA aggregation
            if fcc.ndim == 2:
                fcc = fcc[:, 0]
            cum_fcc = float(np.sum(fcc[SL]))           # total discounted FC spend

            if cum_fcc <= 0:
                # No FC expenditure → threshold undefined
                rows.append(dict(Region=reg, Scenario=scen_label,
                                 GDP_per_cap=round(gdp),
                                 ICER_at_base=np.nan,
                                 P_thresh_1x="N/A",
                                 P_thresh_05x="N/A",
                                 Note="No FC expenditure"))
                continue

            # ── N_fc: discounted FC person-years per dollar of drug price ─
            N_fc = cum_fcc / P0

            # ── Cumulative direct costs and DALYs ────────────────────────
            bl_dirc = econ[bl_dirc_key]
            cs_dirc = econ[cs_dirc_key]
            bl_daly = econ[bl_daly_key]
            cs_daly = econ[cs_daly_key]

            # Flatten if 2-D (PSA arrays vs central arrays differ)
            def _s(arr):
                return float(np.sum(arr[SL, 0] if arr.ndim == 2 else arr[SL]))

            cum_bl_dirc = _s(bl_dirc)
            cum_cs_dirc = _s(cs_dirc)
            cum_bl_daly = _s(bl_daly)
            cum_cs_daly = _s(cs_daly)

            delta_daly    = cum_bl_daly - cum_cs_daly   # DALYs averted (positive)
            delta_dirc    = cum_cs_dirc - cum_bl_dirc   # incremental direct cost
            delta_nonfc   = delta_dirc  - cum_fcc        # non-FC incremental direct cost

            # ── Verify ICER at base price (cross-check against stored value) ─
            if delta_daly > 0:
                icer_check = (delta_nonfc + N_fc * P0) / delta_daly
            else:
                icer_check = np.nan

            # ── Threshold prices ─────────────────────────────────────────
            def _thresh(wtp):
                if delta_daly <= 0 or N_fc <= 0:
                    return np.nan
                p = (wtp * delta_daly - delta_nonfc) / N_fc
                return p   # can be negative → FC cannot be cost-effective

            p1x  = _thresh(wtp_1x)
            p05x = _thresh(wtp_05x)

            def _fmt(p):
                if np.isnan(p):
                    return "N/A"
                if p <= 0:
                    return "Not achievable"
                return f"${p:,.0f}"

            rows.append(dict(
                Region       = reg,
                Scenario     = scen_label,
                GDP_per_cap  = round(gdp),
                ICER_at_base = round(icer_check) if not np.isnan(icer_check) else np.nan,
                P_thresh_1x  = _fmt(p1x),
                P_thresh_05x = _fmt(p05x),
                Note         = ""
            ))

    df = pd.DataFrame(rows, columns=[
        "Region", "Scenario", "GDP_per_cap",
        "ICER_at_base", "P_thresh_1x", "P_thresh_05x", "Note"
    ])
    return df


# Step 3: Run the analysis
region_econ_map = {
    "AFR":    afr_econ,
    "AMR":    amr_econ,
    "EMR":    emr_econ,
    "EUR":    eur_econ,
    "SEAR":   sear_econ,
    "WPR":    wpr_econ,
    "Global": global_econ,
}

price_thresh_df = price_threshold_analysis(
    region_econ_map,
    cost_path = wd + "cost and agg data/costs.xlsx",
    fc_base_price = 5000,
    start_idx = 36,
    end_idx   = 110
)

# Step 4: Print results
print("\n" + "=" * 80)
print("FC Drug Price Threshold Analysis")
print("Maximum per-course price at which ICER ≤ regional WTP threshold")
print(f"(Base-case price = $5,000 | 2026–2099, discounted at 1.5%/yr)")
print("=" * 80)
print(price_thresh_df.to_string(index=False))
print("=" * 80)

# Step 5: Save to Excel with formatting
with pd.ExcelWriter("Table_PriceThreshold.xlsx", engine="xlsxwriter") as writer:
    price_thresh_df.to_excel(writer, sheet_name="Price Thresholds", index=False)
    wb  = writer.book
    ws  = writer.sheets["Price Thresholds"]

    # Column widths
    col_widths = [8, 16, 12, 14, 16, 16, 20]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    # Header format
    hdr_fmt = wb.add_format({
        "bold": True, "bg_color": "#2C3E50", "font_color": "white",
        "align": "center", "valign": "vcenter", "border": 1
    })
    headers = ["Region", "Scenario", "GDP/cap (US$)",
               "ICER @ $5k (US$/DALY)", "Max price\n1× GDP/cap",
               "Max price\n0.5× GDP/cap", "Note"]
    for col, h in enumerate(headers):
        ws.write(0, col, h, hdr_fmt)

    # Alternating row colours; highlight "Not achievable" in red
    fmt_even    = wb.add_format({"bg_color": "#EBF5FB", "border": 1})
    fmt_odd     = wb.add_format({"bg_color": "#FFFFFF",  "border": 1})
    fmt_na      = wb.add_format({"bg_color": "#FADBD8", "border": 1, "font_color": "#922B21"})
    fmt_ok_1x   = wb.add_format({"bg_color": "#D5F5E3", "border": 1, "font_color": "#1E8449"})

    for row_idx, row in price_thresh_df.iterrows():
        base_fmt = fmt_even if row_idx % 2 == 0 else fmt_odd
        for col_idx, val in enumerate(row):
            cell_fmt = base_fmt
            if col_idx in (4, 5) and isinstance(val, str):
                if "Not achievable" in val:
                    cell_fmt = fmt_na
                elif val.startswith("$") and col_idx == 4:
                    # 1× threshold exists → green
                    cell_fmt = fmt_ok_1x
            ws.write(row_idx + 1, col_idx, str(val) if val != "" else "", cell_fmt)

    # Add explanatory note below the table
    note_fmt = wb.add_format({"italic": True, "font_color": "#555555", "font_size": 9})
    note_row = len(price_thresh_df) + 3
    ws.write(note_row, 0,
             "Notes: ICER = incremental cost-effectiveness ratio; WTP = willingness-to-pay; "
             "GDP/cap = regional GDP per capita (2025 US$) from costs.xlsx emp_gdp_lex sheet. "
             "P_thresh is the maximum FC drug price per course at which cumulative ICER (2026–2099) "
             "remains ≤ the stated WTP multiple. Derived analytically from existing model outputs "
             "without additional simulation runs: P_thresh = (WTP × ΔDALYs − ΔDirect_non_FC) / N_fc, "
             "where N_fc = cumulative discounted FC treatment-years at base price $5,000.",
             note_fmt)

print("\n[Price threshold table saved to: Table_PriceThreshold.xlsx]")