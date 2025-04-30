libraryDependencies ++= Seq(
  jdbc,
  evolutions,
  "mysql" % "mysql-connector-java" % "8.0.33",
  "com.typesafe.play" %% "play-slick" % "5.0.0",
  "com.typesafe.play" %% "play-slick-evolutions" % "5.0.0"
)
