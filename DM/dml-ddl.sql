CREATE DATABASE IF NOT EXISTS Sales;


CREATE TABLE IF NOT EXISTS Employee (
	ENr NUMERIC PRIMARY KEY,
    Name VARCHAR(20)
);

ALTER TABLE Employee RENAME COLUMN Name TO EName;

CREATE TABLE IF NOT EXISTS Customer (
		CNR Numeric  primary KEY,
        CName VARCHAR(15)
);

ALTER TABLE Customer ADD COLUMN (ENr Numeric);

CREATE TABLE IF NOT EXISTS Project (
	PNr numeric PRIMARY KEY,
    PName VARCHAR(50),
    CNr Numeric,
    ENr Numeric
    
);

CREATE TABLE IF NOT EXISTS Task (
	PNr numeric,
    ENr numeric,
    Task VARCHAR(50),
    Share numeric
);

ALTER TABLE Customer
ADD FOREIGN KEY (ENr) REFERENCES Employee(ENr);


ALTER TABLE Project
ADD FOREIGN KEY (ENr) REFERENCES Employee(ENr);


-- SHOW CREATE TABLE Customer;

ALTER TABLE Project
DROP foreign key ENr;

ALTER TABLE Project
ADD FOREIGN KEY (ENr) REFERENCES Employee(ENr);

ALTER TABLE Project
ADD FOREIGN KEY (CNr) REFERENCES Customer(CNr);

ALTER TABLE Task
ADD FOREIGN KEY (ENr) REFERENCES Employee(ENr);


