********************************************************************************
* Fix import errors                                                            *
* Last modified Oct 17, 2024                                                   *
********************************************************************************

* Identify and fix instances where a single data field erroniously read into multiple rows.
gen flag_ln_error = 0
gen last_char = substr(v1,-1,.)
replace flag_ln_error = 1 if  last_char != ">" 
replace flag_ln_error = 1 if last_char[_n-1] != ">" & _n > 1
qui: sum flag_ln_error
local s = r(sum)
if `s' > 0 {
	gen flag_first = 1 if flag_ln_error == 1 & last_char[_n - 1] == ">"
	gen line_id = sum(flag_first) if flag_ln_error == 1

	frame put if line_id != ., into(long_data_ln_error)
	frame change long_data_ln_error

	sort obs_num
	bysort line_id (obs_num): gen temp = v1[1] if flag_ln_error == 1
	by line_id: replace temp = temp[_n-1] + v1 if _n > 1 & flag_ln_error == 1
	by line_id: replace temp = temp[_N] if flag_ln_error == 1
	drop if line_id == line_id[ _n - 1 ]
	keep obs_num temp row_id
	rename temp v1
	save `"${DIR}\temp\temp_line_split"', replace

	frame change long_data
	drop if line_id != .
	drop flag_ln_error last_char flag_first line_id   
	frame drop long_data_ln_error
	append using `"${DIR}\temp\temp_line_split"' 
	erase `"${DIR}\temp\temp_line_split.dta"'
	sort obs_num
}
else {
	drop last_char flag_ln_error
}