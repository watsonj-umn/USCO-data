********************************************************************************
* Transform from long-form to wide-form                                        *
* Last modified April 10, 2024                                                 *
********************************************************************************

keep row_id var_nm v_
qui sum row_id
global per_itt = round(1000000 / (_N/r(max)))
global tot_itt = ceil(r(max)/$per_itt) 
capture frame drop hold
frame create hold
timer clear 1
forvalues i = 1/$tot_itt {
	timer on 1
	frame change default
	capture frame drop temp
	global lb = (`i' - 1) * $per_itt + 1
	global ub = `i' * $per_itt
	frame put if row_id >= $lb & row_id <= $ub , into(temp)
	drop if row_id >= $lb & row_id <= $ub
	frame change temp
	reshape wide v_ , i(row_id) j(var_nm) string
	frame change hold
	frameappend temp
	dis "Done with `i' of $tot_itt"
	timer off 1
	timer list 1
}
frame drop default
frame rename hold default