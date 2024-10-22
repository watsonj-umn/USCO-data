********************************************************************************
* Name variables                                           *
* Last modified Oct 17, 2024                                                   *
********************************************************************************

global start = 1 + $record_count
global finish = min( $start + $rec_per_it - 1 , $max_records )
global start = row_id[ $start ]
global finish = row_id[ $finish ]	
frame put if inrange( row_id , $start, $finish) , into(long_data_temp) 
frame change long_data_temp
global record_count = $record_count + _N

replace v1 = trim(v1)
gen a=strpos(v1,">") + 1
gen b=strrpos(v1,"<")
gen v13 = substr(v1, a , b-a ) if b > a
replace v13 = trim(v13)
gen v12 = substr(v1, 2, a-3 )
drop a b 

gen temp_df_1 = _n if strpos(v1, "datafield tag") != 0
replace temp_df_1 = temp_df_1[_n-1] if temp_df_1 >= .
gen temp_df_2 = v12[temp_df_1] if temp_df_1 != .

gen temp_sf_1 = 1 if strpos(v1, "subfield code") != 0
gen var_nm = v12
replace var_nm = temp_df_2 + " " + v12 if temp_sf_1 == 1
capture drop sf
gen sf = subinstr(v12, "subfield code=","",1) if temp_sf_1 == 1
replace sf = trim(subinstr(sf, `"""',"",.)) if temp_sf_1 == 1

rename v13 v_
tostring row_id, gen(temp_dt)
replace v_ = temp_dt if var_nm == "record" | var_nm == "/record"
keep var_nm row_id v_ obs_num sf

replace var_nm = subinstr(var_nm, `"datafield tag=""',"df_",1)
replace var_nm = subinstr(var_nm, `"ind1=" ""',`"ind1="na""',1)
replace var_nm = subinstr(var_nm, `"ind2=" ""',`"ind2="na""',1)
replace var_nm = subinstr(var_nm, `"" ind1=""',"_i1_",1)
replace var_nm = subinstr(var_nm, `"" ind2=""',"_i2_",1)
replace var_nm = subinstr(var_nm, `"subfield code=" ""',`"subfield code="MISS""',1)
replace var_nm = subinstr(var_nm, `"" subfield code=""',"_sf_",1)
replace var_nm = subinstr(var_nm, `"""',"",1) if strpos(var_nm, "df_") == 1
replace var_nm = subinstr(var_nm, `"controlfield tag=""',"cf_",1)
replace var_nm = subinstr(var_nm, `"""',"",1) if strpos(var_nm, "cf_") == 1
replace var_nm = "record_end" if var_nm == "/record"
replace var_nm = "record_beg" if var_nm == "record"
drop if var_nm == "/datafield"

* Some datafields are repeatable within a single observation. Datafields with 
* multiple entries are labeled with the sufix "_#". 
gen double org_order = _n
sort row_id var_nm org_ord
bysort row_id var_nm : gen temp = [_n]
qui sum temp
global max_reps = r(max)
tostring temp, replace
gen reps = temp
replace var_nm = var_nm + "_" + temp
sort org_order
drop org_order temp

drop if v_ == ""
drop if v_ == " "

** fix case typos in subfield identifiers
gen temp_1 = lower(var_nm)
gen temp_2 = 1 if temp_1 != var_nm
qui: sum temp_2
local n_drop = r(sum)
global field_drop_count = $field_drop_count + `n_drop'
drop if temp_2 == 1

drop temp*

replace var_nm = strtrim(var_nm)
gen field_number = substr(var_nm, 4, 3) if substr(var_nm, 1, 2) == "df"

frame change temp_large
frameappend long_data_temp
*****
gen temp = var_nm if sf != ""
split temp, p("_") l(6)
rename temp4 i1
rename temp6 i2
rename temp1 field_type
capture drop temp*
drop if var_nm == "record_beg_1" | var_nm == "record_end_1"
*replace field_number = "000" if var_nm == "leader_1"
destring field_number, generate(fn) force


sort row_id fn sf reps
drop reps
gen reps = 1
replace reps = reps[_n-1] + 1 if fn == fn[_n-1] & sf == sf[_n-1] & row_id == row_id[_n-1]
tostring reps , replace
gen hold_var_nm = "df_" + field_number + "_sf_" + sf +"_i1_"+i1+"_i2_"+i2+"_rep_"+reps if field_type == "df"
keep row_id v_ var_nm obs_num field_number hold_var_nm

****
if ${i}/10 == ceil(${i}/10) {
	compress
	save `"${DIR}\temp\temp_${filecount}"', replace
	clear
	global filecount = 1 + $filecount
	frame change long_data
	frame drop long_data_temp	
}
else if ${i} == $max_itts {
	compress
	save `"${DIR}\temp\temp_${filecount}"', replace
	clear
	frame change long_data
	frame drop long_data_temp
}
else {
	frame change long_data
	frame drop long_data_temp
}

global i = $i + 1