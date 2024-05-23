package org.example

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{avg, col, collect_list, collect_set, count, expr, first, isnan, isnotnull, max, regexp_replace, split, to_date, udf}

object SparkChallenge {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[1]")
      .appName("Spark Challenge")
      .getOrCreate()

    // Read and clean data
    val apps_df = clean_data(read_csv(spark, "src/main/resources/data/googleplaystore.csv"))
    val user_reviews_df = clean_data(read_csv(spark, "src/main/resources/data/googleplaystore_user_reviews.csv"))

    // PART 1
    val df_1 = user_reviews_df
      .filter(!isnan(col("Sentiment_Polarity")))
      .groupBy("App")
      .agg(avg("Sentiment_Polarity").alias("Average_Sentiment_Polarity"))
      .na.fill(0, Seq("Average_Sentiment_Polarity"))

    // PART 2
    val df_2 = apps_df
      .filter(!isnan(col("Rating")) && col("Rating") >= 4.0)
      .filter(isnotnull(col("Rating")))
      .sort(col("Rating").desc)

    write_csv(spark, df_2, "src/main/resources/data/output/part2", "best_apps.csv")

    // PART 3
    val dollarsToEuros = udf((dollars: String) => dollars.replaceAll("\\$", "").toDouble * 0.92)

    val size_in_mb = udf((size: String) => {
      size.takeRight(1) match {
        case "M" => size.dropRight(1).toDouble
        case "k" => size.dropRight(1).toDouble / 1024.0
        case _ => 0.0
      }
    })

    val df_3 = apps_df
      .groupBy("App")
      .agg(
        collect_set("Category").as("Categories"),
        max("Reviews").cast("long").as("Reviews")
      )
      .join(apps_df.select(
        "App",
        "Rating",
        "Size",
        "Installs",
        "Type",
        "Price",
        "Content Rating",
        "Genres",
        "Last Updated",
        "Current Ver",
        "Android Ver",
      ), Seq("App"), "inner")
      .withColumn("Rating", col("Rating").cast("double"))
      .withColumn("Size", size_in_mb(col("Size")))
      .withColumn("Price", dollarsToEuros(col("Price")))
      .withColumn("Genres", split(col("Genres"), ";"))
      .withColumn("Last Updated", to_date(col("Last Updated")).as("Last_Updated"))
      .withColumnRenamed("Current Ver", "Current_Version")
      .withColumnRenamed("Android Ver", "Minimum_Android_Version")

    // PART 4
    val df_4 = df_3.join(df_1, Seq("App"))
    write_gzip(spark, df_4, "src/main/resources/data/output/part4", "googleplaystore_cleaned")

    // PART 5
    val df_5 = df_4
      .groupBy("Genres")
      .agg(
        count("App").as("Count"),
        avg("Rating").as("Average_Rating"),
        avg("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity")
      )

    write_gzip(spark, df_5, "src/main/resources/data/output/part5", "googleplaystore_metrics")

    spark.stop()
  }

  def read_csv(sparkSession: SparkSession, csv_path: String): DataFrame = {
    sparkSession.read
      .option("header", value = true)
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .csv(path = csv_path)
  }

  def clean_data(df: DataFrame): DataFrame = {
    df
      .dropDuplicates()
      .withColumn("App", regexp_replace(col("App"), "[\"\\-: ]", ""))
  }

  def write_csv(spark_session: SparkSession, df: DataFrame, output_path: String, file_name: String): Unit = {
    df.write
      .option("header", value = true)
      .option("sep", "§")
      .mode(saveMode = "overwrite")
      .csv(output_path)

    val fs = FileSystem.get(spark_session.sparkContext.hadoopConfiguration)
    val srcPath = fs.globStatus(new Path(output_path + "/part*.csv"))(0).getPath
    val destPath = new Path(output_path + "/" + file_name)
    fs.rename(srcPath, destPath)
  }

  def write_gzip(spark_session: SparkSession, df: DataFrame, output_path: String, file_name: String): Unit = {
    df.write
      .option("compression", "gzip")
      .mode(saveMode = "overwrite")
      .parquet(output_path)

    val fs = FileSystem.get(spark_session.sparkContext.hadoopConfiguration)
    val srcPath = fs.globStatus(new Path(output_path + "/part*.parquet"))(0).getPath
    val destPath = new Path(output_path + "/" + file_name)
    fs.rename(srcPath, destPath)
  }
}