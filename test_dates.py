from datetime import timedelta, date, datetime


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


# fill in zero values
start_date = datetime.strptime('2020-04-16', "%Y-%m-%d").date()
end_date = date.today()

for single_date in daterange(start_date, end_date):
    test = single_date.strftime("%Y-%m-%d")