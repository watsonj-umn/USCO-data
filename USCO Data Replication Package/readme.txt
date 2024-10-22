COPYRIGHT DATA PROCESSING 
*************************

Notes:
• Created using Stata 17. Stata 16 or later is required for .do files to run. 
• The scripts are in beta testing and are not guaranteed to produce accurate results. Individual users should verify results for themselves. 
• Questions or comments can be directed to blutes@copyright.gov.
• Corresponding Author: Brent Lutes

Overview
********
The scripts perform three primary tasks: Import and parse the raw MARC21-format XML data, clean and standardize the data, and create tabular datasets organized by type of work or type of record.

File "01" and its subroutines import the raw MARC21-format XML data, parse the XML tags, fix data structure errors, and then sort the observations into files based on the type or work or the type of record. This version of the data is unabridged and stored in long form. The output (stored in "{DIR}/Processed Data/Long form data") has three columns:
	1) "row_id" contains a unique ID for each record in the file.  
	2) "v_” contains the value for each variable, which can be string or numeric. 
	3) "var_nm" contains the name of each variable, according to the MARC 21 naming convention. 

There are three types of information provided in the data, corresponding to three types of variables:
	1) "Leader" is an alphanumeric code that provides information for the processing of the record, as laid out in the Marc 21 documentation. Leader fields are labeled "leader".
	2) "Control Fields" provide identifiers and other coded information about the record. Specific control fields are identified by a three-digit number, corresponding to the Marc 21 documentation. Each control field may have subfields (although this is uncommon). Control fields are identified with a "cf" prefix, followed by, 1) a three-digit code indicating which control field it is, and 2) an alphanumeric subfield code, when applicable. For example, the variable name for control field 005 is "cf_005".
	3) "Data Fields" contain data entries associated with the record (e.g., the title of a work). Specific data fields are identified by a three-digit code, corresponding to the Marc 21 documentation. Each data field may (and typically does) have subfields, and each subfield has two alphanumeric "indicators". Certain fields are repeatable, thus requiring an indicator for repetitions. Data fields are identified with a "df" prefix, followed by, 
		a) a three-digit code indicating which data field it is, 
		b) an alphanumeric subfield ("sf") code, when applicable, 
		c) indicator-1, 
		d) indicator-2, and 
		e) a value indicating the repetition (again noting that some data fields are repeatable). 
	For example, the variable name for the third repetition of data field 245, subfield z, with indicator-1 = a, and indicator-2 = na, is "df_sf_z_i1_a_i2_na_3".
For more information about MARC21 data see https://www.loc.gov/marc/bibliographic/bdintro.html, or https://www.copyright.gov/policy/women-in-copyright-system/LOC-Copyright-Data-as-Distributed-in-the-MARC%2021-Format.pdf

File "02" and its subroutines take the long form data created by file "01", abridge them, clean them, map MARC21 datafields into fields with descriptive labels, and transform the data into wide form. For details on how the MARC21 datafields are mapped into cohesive descriptive fields review the "02.*" series of scripts. The Data are abridged in two ways. First, certain datafields are omitted. These are datafields that are either seldom populated or are otherwise trivial or redundant. For example, copyright registration numbers can appear in various formats across several datafields; these are combined into one cohesive datafield. The second way the data are abridged is by limiting the number of times a datafield can be repeated in a single record to 10. This pertains to datafields related to authors and claimants. A very small portion of records have hundreds of authors and claimants, which would translate into thousands of columns, making the storage and manipulation of wide-form data difficult. Limiting authors and claimants to 10 each resolves this problem at the cost of a small portion of the records being incomplete (the constraint on authors is binding for less than 0.1% of records and the constraint on claimants is binding for less than 0.003% of records). Note, however, that this is more impactful for certain types of records than others (for example, compilation albums often have many recording artists and songwriters associated with them). If users believe this constraint will have a nontrivial effect on their analysis it is recommended to use the unabridged long-form data instead.
 

Procedure
*********
Steps for parsing text in raw data and create .dta files with data in long form.
1)	Unzip the "copyright_data_replication_pckg.zip into an empty directory.
2)	Download and unzip raw data into "/Raw Data" folder. For raw data go to https://copyright.gov/economic-research/usco-datasets/  and select "Raw Unparsed Uncategorized XML" or initiate download using https://data.copyright.gov/Raw%20unparsed%20uncategorized%20XML/USCO%20public%20data.zip
3)	Open 00.00._Master.do. 
4)	Enter root directory (ending with "\Data replication package") on line 17.
5)	Scripts can be tested on a small subset of data by entering "TRUE" on line 20 (global test_run = "TRUE"). For processing full data enter "FALSE".
6)	Execute
