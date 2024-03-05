# Credit-Card-Churn-Prediction

## Background and Context
The Thera bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.

Customers’ leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and reason for same – so that bank could improve upon those areas

## Objective
The objective is to explore and visualize the dataset, build a classification model to predict if the customer is going to churn or not, optimize the model using appropriate techniques, and generate a set of insights and recommendations that will help the bank.

## Data Dictionary
CLIENTNUM: Client number. Unique identifier for the customer holding the account

Attrition_Flag: Internal event (customer activity) variable - if the account is closed then "Attrited Customer" (1), else "Existing Customer" (0)

Customer_Age: Age in Years

Gender: Gender of the account holder

Dependent_count: Number of dependents

Education_Level: Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to a college student), Post-Graduate, Doctorate.

Marital_Status: Marital Status of the account holder

Income_Category: Annual Income Category of the account holder

Card_Category: Type of Card

Months_on_book: Period of relationship with the bank

Total_Relationship_Count: Total no. of products held by the customer

Months_Inactive_12_mon: No. of months inactive in the last 12 months

Contacts_Count_12_mon: No. of Contacts between the customer and bank in the last 12 months

Credit_Limit: Credit Limit on the Credit Card

Total_Revolving_Bal: The balance that carries over from one month to the next is the revolving balance

Avg_Open_To_Buy: Open to Buy refers to the amount left on the credit card to use (Average of last 12 months)

Total_Trans_Amt: Total Transaction Amount (Last 12 months)

Total_Trans_Ct: Total Transaction Count (Last 12 months)

Total_Ct_Chng_Q4_Q1: Ratio of the total transaction count in 4th quarter and the total transaction count in 1st quarter

Total_Amt_Chng_Q4_Q1: Ratio of the total transaction amount in 4th quarter and the total transaction amount in 1st quarter

Avg_Utilization_Ratio: Represents how much of the available credit the customer spent

## Business Recommendations

Factors that drive the attrition - Total_Trans_Ct, Total_Revolving_Bal, Total_Trans_Amt, Total_Relationship_Count

Total_Trans_Ct: Less number of transactions in a year leads to attrition of a customer - to increase the usage of cards the bank can provide offers like cashback, special discounts on the purchase of something, etc so that customers feel motivated to use their cards.

Total_Revolving_Bal: Customers with less total revolving balance are the ones who attrited, such customers must have cleared their dues and opted out of the credit card service. After the customer has cleared the dues bank can ask for feedback on their experience and get to the cause of attrition.

Total_Trans_Amt: Less number of transactions can lead to less transaction amount and eventually leads to customer attrition - Bank can provide offers on the purchase of costlier items which in turn will benefit the customers and bank both.

Total_Relationship_Count: Attrition is highest among the customers who are using 1 or 2 products offered by the bank - together they constitute ~55% of the attrition - Bank should investigate here to find the problems customers are facing with these products, customer support, or more transparency can help in retaining customers.

Female customers should be the target customers for any kind of marketing campaign as they are the ones who utilize their credits, make more and higher amount transactions. But their credit limit is less so increasing the credit limit for such customers can profit the bank.

Months_Inactive: As inactivity increases the attrition also increases, 2-4 months of inactivity are the biggest contributors of attrition -Bank can send automated messages to engage customers, these messages can be about their monthly activity, new offers or services, etc.

Highest attrition is among the customers who interacted/reached out the most with/to the bank, This indicates that the bank is not able to resolve the problems faced by customers leading to attrition - a feedback collection system can be set up to check if the customers are satisfied with the resolution provided, if not, the bank should act upon it accordingly.

Thera Bank should target customers who transactioned the amount of around 4,404 dollars in the last 12 months. This should be looked into as the average annual income for most customers are less than 40k dollars.

The total transaction amount and count are strong negative factors for customers. The bank should provide reward system for depositing "x" amount of money in their account.

The relationship between the bank and its customers is 35 months or almost 3 years on average. At least 75% of customers were inactive for 3 months in the last 12 months. At least 25% or more customers had 2 contacts between the customer and bank in the last 12 months. To increase the relationship, the bank could provide more options in their customer service, such as options for lower late fees if the customer is late for paying.

The best test recall is ~90% and the test precision is 89% at the same time. This means that the model is good at identifying non-defaulter, therefore, the bank has more chances of not losing its customers since it was identified at the right time.
