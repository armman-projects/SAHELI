
## Socio-demographic features
Data Corresponding to every beneficiary registered in the mMitra program
- enroll_gest_age: Gestation Age at Time of Enrollment
- registration_date: Date of Registration with the mMitra Program
- entry_date: Date of entry into the database
- language: Preferred language for Automated Voice Calls
- age: Beneficiary’s Age
- education: Beneficiary’s Education. This is categorical variable representing one of 6 education levels- ['Illiterate', '1 - 5', '6 - 9', '10 Pass', '12 Pass', 'Graduate', 'Post Graduate']
- phone_owner: Who owns the phone registered by the beneficiary. One of 3 categorical variables - Mother, Husband, Family
- call_slots: The preferred time slot for receiving the automated voice call
- enroll_delivery_status: A binary flag which represents whether the beneficiary has already delivered the baby at the time of registration
- ChannelType: The channel through which the beneficiary is registered into the ARMMAN program. Can be one of 3 categorical values - Hospital, Door-to-Door registration by ARMMAN health workers, Door-to-Door registration by partner NGO health workers.
- income_bracket: Monthly family income in Indian Rupees (INR). Categorical variable representing buckets ['0-5k', '5k-10k', '10k-15k', '15k-20k', '20k-25k', '25k-30k', '>30k']
- ngo_hosp_id: The hospital ID at which the beneficiary is registered.
- g: Gravidity
- p: Parity
- s: Number of past still-births
- l: Number of past live-births


## Call data features:
Data corresponding to every automated call sent to a beneficiary.

- user_id: A unique ID assigned to every beneficiary
- startdatetime: The timestamp of when the call is sent
- duration: The duration of the call
- gest_age: The gestational age of the beneficiary when the call is received
- callStatus: The status of call connection. Contains multiple flags provided by the telecom provider such as - Call Successful, Network Busy, Phone Switched off, Invalid number.
- dropreason: The reason for the call end.
- media_id: A unique ID representing the content of message that is played in the automated call

## Current training data details:
- Number of beneficiaries served with SAHELI until now: 98K
- Number of beneficiaries enrolling every month : 20k beneficiaries per month 
- Number of interventions: 1000 interventions per week
- Warmup period: The time period between registration of a beneficiary and when we start delivering service calls. For SAHELI, we keep it at 6 weeks.
