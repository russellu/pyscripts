# mortgage calculator

ask = 184500
interest_rate = 0.028/12
gross_rent = 19104
percent_down = 0.2
n_payments = 12*25
principle = ask*percent_down

payment = (interest_rate*(ask-ask*percent_down)) / (1-(1+interest_rate)**-n_payments)
print(payment)

after_mtg = gross_rent - payment*12

roi = (after_mtg - 5000)/principle





