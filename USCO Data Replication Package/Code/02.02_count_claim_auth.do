********************************************************************************
* Count authors and claimants                                                  *
* Last modified April 10, 2024                                                 *
********************************************************************************

gen double order = _n
egen total_prty = max(rep) , by(row_id field)
gen flag = 1 if rep == total_prty & field == "df_249_sf_c"
expand 2 if flag == 1, gen(new)
sort order new
capture drop temp
tostring total_prty, gen(temp)
replace v_ = temp if new == 1
replace var_nm = "num_claimants" if new == 1
replace rep = 1  if new == 1
drop flag new 
gen flag = 1 if rep == total_prty & field == "df_279_sf_c"
expand 2 if flag == 1, gen(new)
sort order new
replace v_ = temp if new == 1
replace var_nm = "num_authors" if new == 1
replace rep = 1  if new == 1
drop flag new temp order total_prty
keep if rep < 11 | (rep < 21 & (field == "df_700_sf_a" | field == "df_710_sf_a" )) | (rep < 31 & field == "df_505_sf_t" )
sort row_id field rep