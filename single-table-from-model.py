
import pandas as pd

appication_train = pd.read_csv("data/application_train.csv")
bureau_balance = pd.read_csv("data/bureau_balance.csv")
bureau = pd.read_csv("data/bureau.csv")
credit_card_balance = pd.read_csv("data/credit_card_balance.csv")
installments_payments = pd.read_csv("data/installments_payments.csv")
POS_CASH_balance = pd.read_csv("data/POS_CASH_balance.csv")
previous_application = pd.read_csv("data/previous_application.csv")

result = appication_train.join(bureau_balance).join(bureau, lsuffix="_a", rsuffix="_b").join(credit_card_balance, lsuffix="_c", rsuffix="_d").join(installments_payments, lsuffix="_f", rsuffix="_g").join(POS_CASH_balance, lsuffix="_h", rsuffix="_i").join(previous_application, lsuffix="_j", rsuffix="_k")
print(result.head())
result.to_csv("single-table-raw.csv", index=False)
