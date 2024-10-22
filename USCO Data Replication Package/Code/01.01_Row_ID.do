********************************************************************************
* Add row ID                                                                   *
* Last modified Oct 17, 2024                                                   *
********************************************************************************

* Identify rows that corrispond to a common record (observations are spread accross multiple rows in original file)
gen double obs_num = _n
gen temp_1 = 0
replace temp_1 = 1 if v1 == "<record>"
gen double row_id = sum(temp_1)
gen double temp_2 = _n if v1 == "</record>"
egen double temp_3 = max(temp_2)
drop if obs_num > temp_3
drop if row_id == 0
drop temp*