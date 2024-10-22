********************************************************************************
* Drop unused datafields                                                       *
* Last modified April 10, 2024                                                 *
********************************************************************************

split var_nm, p("rep_")
destring var_nm2, gen(rep) force
drop var_nm1 var_nm2
capture drop field
gen field = substr(var_nm, 1, 11)
capture drop flag
gen flag = 0
replace flag = 1 if field == "df_017_sf_a" | field == "df_017_sf_c" | field == "df_017_sf_e" | field == "df_017_sf_g" | field == "df_017_sf_n" | field == "df_017_sf_r" | field == "df_020_sf_a" | field == "df_022_sf_a" | field == "df_027_sf_a" | field == "df_028_sf_b" | field == "df_046_sf_c" | field == "df_046_sf_k" | field == "df_046_sf_m" | field == "df_242_sf_a" | field == "df_245_sf_a" | field == "df_245_sf_c" | field == "df_246_sf_a" | field == "df_249_sf_c" | field == "df_249_sf_u" | field == "df_257_sf_a" | field == "df_260_sf_a" | field == "df_260_sf_b" | field == "df_260_sf_c" | field == "df_263_sf_a" | field == "df_263_sf_b" | field == "df_263_sf_c" | field == "df_917_sf_a" | field == "df_269_sf_c" | field == "df_269_sf_d" | field == "df_279_sf_c" | field == "df_279_sf_d" | field == "df_279_sf_e" | field == "df_279_sf_f" | field == "df_279_sf_g" | field == "df_279_sf_h" | field == "df_279_sf_p" | field == "df_310_sf_a" | field == "df_349_sf_a" | field == "df_440_sf_a" | field == "df_440_sf_v" | field == "df_500_sf_a" | field == "df_505_sf_t" | field == "df_508_sf_a" | field == "df_549_sf_a" | field == "df_589_sf_a" | field == "df_591_sf_a" | field == "df_596_sf_a" | field == "df_597_sf_a" | field == "df_700_sf_a" | field == "df_700_sf_d" | field == "df_710_sf_a" | field == "df_700_sf_e" | field == "df_710_sf_e" | field == "df_740_sf_a" | field == "df_779_sf_b" | field == "df_779_sf_n" | field == "df_779_sf_o" | field == "df_917_sf_b" | field == "df_917_sf_f" | field == "df_917_sf_p" | field == "df_017_sf_d" | field == "df_017_sf_f" | field == "df_017_sf_h" | field == "df_779_sf_q"
replace flag = 1 if regexm(var_nm, "df_269_sf_")
replace flag = 1 if regexm(var_nm, "df_279_sf_")
replace flag = 1 if regexm(var_nm, "df_249_sf_")
keep if flag == 1
drop flag
tostring rep, gen(rt)
replace var_nm = field + "_rep_" + rt
replace var_nm = "reg_num" if field == "df_017_sf_a" | field == "df_027_sf_a" | field == "df_917_sf_a"
replace var_nm = "title" if field == "df_245_sf_a"  | field == "df_240_sf_a"  | field == "df_242_sf_a"
collapse (firstnm) v_ rep field , by(var_nm row_id)
sort row_id field rep
capture drop cnt
gen cnt = 0
replace cnt = 1 if rep == 1
capture drop freq
egen freq = sum(cnt) , by(field)
qui sum row_id
local max = r(max)
replace freq = freq / `max'
*keep if freq > .1 | field == "df_700_sf_d" 