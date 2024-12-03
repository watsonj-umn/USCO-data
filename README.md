# USCO-data

## Usage

If you do not wish to re-run the record linking process, simply
download the file *crsp_compustat_regdata_1978-2021.csv* that links Compustat/CRSP identifiers to copyright
registrations and federal litigation.

Otherwise, run link.sh:

```shell
$ ./link.sh
```


## Codebook
| Variable Name | Description                                                             |
|--------------|-------------------------------------------------------------------------|
| LPERMCO      | CRSP PERMCO Identifier                                                  |
| fyear        | Calendar Year                                                           |
| gvkey        | Compustat GVKEY Identifier                                             |
| LPERMNO      | CRSP PERMNO Identifier                                                 |
| reg_count    | Count of US Copyright registrations linked to LPERMCO in fyear         |
| plt_count    | Count of litigation linked to LPERMCO as plaintiff, filed in fyear     |
| def_count    | Count of suits linked to and filed against LPERMCO (as defendant) in fyear |


## Configuration
 * To re-run linking code, download input files into raw-data/ 
   * cv88on.txt: Federal Judicial Center Data
     * Available at https://www.fjc.gov/sites/default/files/idb/textfiles/cv88on.zip (6/1/2021 version)
   * reg_*.dta: Tabular US Copyright office records
     * Available at
       https://copyright.gov/economic-research/usco-datasets/
       (2/20/2024  version)
   * crsp_link.csv: CRSP/Compustat Merged Database - Linking Table
     (4/15/2024 version). 
     * Available on WRDS under ccmlinktable, save all variables with
       gvkey identifiers using LU and LC linking options. 
   * compustat_all.csv: Compustat Daily Updates - Fundamentals Annual
     (2/21/2024 version)
	 *  Available on WRDS under comp_na_daily_all, save all
        variables. Screening Variables: Consolidation Level (C),
        Industry Format (INDL, FS), Data Format (STD), Population
        Source (D), Currency (USD, CAD), Company Status (A,I).

	 


