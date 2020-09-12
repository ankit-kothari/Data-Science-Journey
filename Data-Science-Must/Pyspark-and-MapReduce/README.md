# tools
- Important Commands
    - Loading a data set into a resilient distributed data set (RDD):

        **`raw_data = sc.textFile("daily_show.tsv")`**

    - Printing out the first five elements of the RDD:

        **`raw_data.take(5)`**

    - Mapping a function to every element in the RDD:

        **`daily_show = raw_data.map(lambda line: line.split('\t'))`**

    - Merging the values for each similar key:

        **`tally = daily_show.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y: x+y)`**

    - Retuning the count of an RDD:

        **`tally.take(tally.count())`**

    - Filtering an RDD:

        **`rdd.filter(lambda x: x % 2 == 0)`**

    - Generate a sequence of values from an RDD:

        **`def hamlet_speaks(line):
        id = line[0]
        speaketh = False
        if "HAMLET" in line: speaketh = True
        if speaketh: yield id,"hamlet speaketh!"
        hamlet_spoken = split_hamlet.flatMap(lambda x: hamlet_speaks(x))`**

    - Return a list representation of an RDD:

        **`hamlet_spoken_lines.collect()`**

    - Return the number of elements in an RDD:

        **`hamlet_spoken_lines.count()`**

    - Instantiating the SQLContext class:

        **`from pyspark.sql import SQLContext
        sqlCtx = SQLContext(sc)`**

    - Reading in JSON data:

        **`df = sqlCtx.read.json("census_2010.json")`**

    - Using the show() method to print the first five rows:

        **`df.show(5)`**

    - Using the head method and a for loop to return the first five rows of the DataFrame:

        **`first_five = df.head(5)
        for r in first_five: print(r.age)`**

    - Using the show method to display columns:

        **`df.select('age', 'males', 'females')`**

    - Converting a Spark DataFrame to a pandas DataFrame:

        **`pandas_df = df.toPandas()`**

    - Registering an RDD as a temporary table:

        **`from pyspark.sql import SQLContext
        sqlCtx = SQLContext(sc)
        df = sqlCtx.read.json("census_2010.json")
        df.registerTempTable('census2010')`**

    - Returning a list of tables:

        **`tables = sqlCtx.tableNames()`**

    - Querying a table in Spark:

        **`sqlCtx.sql('select age from census2010').show()`**

    - Calculating summary statistics for a DataFrame:

    **`query = 'select males,females from census2010' sqlCtx.sql(query).describe().show()`**

## Important Terminologies

### MapReduce

MapReduce efficiently distribute calculations over hundreds or thousands of computers to calculate the result in parallel. Hadoop is an open source project that quickly became the dominant processing toolkit for big data.

### Hadoop

- Hadoop consists of a file system (Hadoop Distributed File System, or HDFS) and its own implementation of the MapReduce paradigm. MapReduce converts computations into Map and Reduce steps that Hadoop can easily distribute over many machines.
- Hadoop made it possible to analyze large data sets, but relied heavily on disk storage (rather than memory) for computation. While it's inexpensive to store large volumes of data this way, it makes it accessing and processing much slower.
- Hadoop wasn't a great solution for calculations requiring multiple passes over the same data or many intermediate steps, due to the need to write to and read from the disk between each step. This drawback also made Hadoop difficult to use for interactive data analysis.
- Hadoop also suffered from suboptimal support for the additional libraries such as SQL and machine learning implementations. Once the cost of RAM (computer memory) started to drop significantly, augmenting or replacing Hadoop by storing data in-memory quickly emerged as an appealing alternative.

### Spark

- Spark, which uses distributed, in-memory data structures to improve speeds for many data processing workloads by several orders of magnitude.
- The core data structure in Spark is a resilient distributed data set (RDD). RDD is Spark's representation of a data set that's distributed across the RAM, or memory, of a cluster of many machines.
- Similar to a pandas DataFrame, we can load a data set into an RDD, and then run any of the methods accessible to that object.

### **PySpark**

- While the Spark toolkit is in Scala, a language that compiles down to byte-code for the JVM,  [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) that allows us to interface with RDDs in Python.

## Practical Application

### SparkContext Managers

- In Spark, the SparkContext object manages the connection to the clusters, and coordinates the running of processes on those clusters. More specifically, it connects to the cluster managers. The cluster managers control the executors that run the computations.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21ff79e8-64a6-483d-b4bb-cd1c72119808/Screen_Shot_2020-06-17_at_11.16.09_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21ff79e8-64a6-483d-b4bb-cd1c72119808/Screen_Shot_2020-06-17_at_11.16.09_AM.png)

- Spark  can run locally on your own computer. Spark will simulate distributing your calculations over many machines by automatically slicing your computer's memory into partitions.
- Spark's RDD implementation also lets us evaluate code "lazily," meaning we can postpone running a calculation until absolutely necessary. The advantage of "lazy" evaluation is that we can build up a queue of tasks and let Spark optimize the overall workflow in the background. In regular Python, the interpreter can't do much workflow optimization.
- We automatically have access to the SparkContext object `sc`. We then run the following code to read the TSV data set into an RDD object `raw_data`:
- The RDD object `raw_data` closely resembles a list of string objects, with one object for each line in the data set. We then use the `take()` method to print the first five elements of the RDD:

```python
raw_data = sc.textFile("daily_show.tsv")
raw_data.take(5)
```

### Data pipelining

- Every operation or calculation in Spark is essentially a series of steps that we can chain together and run in succession to form a pipeline. Each step in the pipeline returns either a Python value (such as an integer), a Python data structure (such as a dictionary), or an RDD object.
- map(function)

apply works on a row / column basis of a DataFrame, applymap works element-wise on a DataFrame, and map works element-wise on a Series

```python
daily_show = raw_data.map(lambda line: line.split('\t'))
daily_show.take(5)

# creating a count by year
tally = daily_show.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y: x+y)
print(tally)
tally.take(tally.count())

def filter_year(line):
    if line[0] == 'YEAR':
        return False
    else:
        return True

#removing the header 
filtered_daily_show = daily_show.filter(lambda line: filter_year(line))

#pipelining adding multiple data transformationsin one go
filtered_daily_show.filter(lambda line: line[1] != '') \
                   .map(lambda line: (line[1].lower(), 1)) \
                   .reduceByKey(lambda x,y: x+y) \
                   .take(5)
```

### **Transformations and Actions**

There are two types of methods in Spark:

- 1. Transformations - map(), reduceByKey(), flatMap()
- 2. Actions - take(), reduce(), saveAsTextFile(), collect(), filter(), count()

Whenever we use an action method, Spark forces the evaluation of lazy code. If we only chain together transformation methods and print the resulting RDD object, we'll see the type of RDD (e.g. a PythonRDD or PipelinedRDD object), but not the elements within it. That's because the computation hasn't actually happened yet.

### Immutablilty

- RDD objects are immutable, meaning that we can't change their values once we've created them. In Python, list and dictionary objects are mutable (we can change their values), while tuple objects are immutable. The only way to modify a tuple object in Python is to create a new tuple object with the necessary updates.

## Iterators, Generators, Yield

[What does the "yield" keyword do?](https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do/231855#231855)

Any function that returns a sequence of data in PySpark (versus a guaranteed Boolean value, like filter() requires) must use a yield statement to specify the values that should be pulled later.

### flatMap()

flatMap() is different than map() because it doesn't require an output for every element in the RDD. The flatMap() method is useful whenever we want to generate a sequence of values from an RDD

## Spark DataFrames

- Spark DataFrames allow us to modify and reuse our existing pandas code to scale up to much larger data sets. They also have better support for various data formats. We can even use a SQL interface to write distributed SQL queries that query large database systems and other data stores.
- The Spark SQL class is very powerful. It gives Spark more information about the data structure we're using and the computations we want to perform. Spark uses that information to optimize processes.
- This class allows us to read in data and create new DataFrames from a wide range of sources. It can do this because it takes advantage of Spark's powerful [Data Sources API](https://databricks.com/blog/2015/01/09/spark-sql-data-sources-api-unified-data-access-for-the-spark-platform.html).
    - File Formats
        - Parquet, Amazon S3 (cloud storage service)
        - JSON, CSV/TSV, XML
    - Big Data Systems
        - Hive, Avro, HBase
    - SQL Database Systems
        - MySQL, PostgreSQL
    - Data science organizations often use a wide range of systems to collect and store data, and they're constantly making changes to those systems. Spark DataFrames allow us to interface with different types of data, and ensure that our analysis logic will still work as the data storage mechanisms change.
    - We can query Spark DataFrame objects with SQL. The SQLContext class gets its name from this capability.

When we read data into the SQLContext object, Spark:

- Instantiates a Spark DataFrame object
- Infers the schema from the data and associates it with the DataFrame
- Reads in the data and distributes it across clusters (if multiple clusters are available)
- Returns the DataFrame object

- Unlike pandas DataFrames, however, Spark DataFrames are immutable, which means we can't modify existing objects.
- Pandas and Spark DataFrames also have different underlying data structures. Pandas DataFrames are built around Series objects, while Spark DataFrames are built around RDDs.

## Spark SQL

- we need to tell Spark to treat the DataFrame as a SQL table. Spark internally maintains a virtual database within the SQLContext object. This object, which we enter as `sqlCtx`, has methods for registering temporary tables.
- To register a DataFrame as a table, call the createOrReplaceTempView() on that DataFrame object. This method requires one string parameter, `name`, that we use to set the table name for reference in our SQL queries.
- Because the results of SQL queries are DataFrame objects, we can combine the best aspects of both DataFrames and SQL to enhance our workflow. For example, we can write a SQL query that quickly returns a subset of our data as a DataFrame.
- The functions and operators from SQLite that we've used in the past are available for us to use in Spark SQL:
    - COUNT()
    - AVG()
    - SUM()
    - AND
    - OR
- One of the most powerful use cases in SQL is joining tables. Spark SQL takes this a step further by enabling you to run join queries across data from multiple file types. Spark will read any of the file types and formats it supports into DataFrame objects and we can register each of these as tables within the SQLContext object to use for querying.

```python
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
df = sqlCtx.read.json("census_2010.json")
df.createOrReplaceTempView('census2010')
df_2000 = sqlCtx.read.json("census_2000.json")
df_1990 = sqlCtx.read.json("census_1990.json")
df_1980 = sqlCtx.read.json("census_1980.json")

df_2000.createOrReplaceTempView('census2000')
df_1990.createOrReplaceTempView('census1990')
df_1980.createOrReplaceTempView('census1980')
tables = sqlCtx.tableNames()
print(tables)

query = """
 select sum(census2010.total), sum(census2000.total), sum(census1990.total)
 from census2010
 inner join census2000
 on census2010.age=census2000.age
 inner join census1990
 on census2010.age=census1990.age
"""
sqlCtx.sql(query).show()

Output:
['census1980', 'census1990', 'census2000', 'census2010', 'census_2010']

+-------+-------+
|  total|  total|
+-------+-------+
|4079669|3733034|
|4085341|3825896|
|4089295|3904845|
|4092221|3970865|
|4094802|4024943|
|4097728|4068061|
|4101686|4101204|
|4107361|4125360|
|4115441|4141510|
|4126617|4150640|
|4137506|4152174|
|4144742|4145530|
|4169316|4139512|
|4220043|4138230|
|4285424|4137982|
|4347028|4133932|
|4410804|4130632|
|4451147|4111244|
|4454165|4068058|
|4432260|4011192|
+-------+-------+
only showing top 20 rows
```
