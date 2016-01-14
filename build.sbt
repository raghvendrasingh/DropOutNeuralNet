import sbt._

name := "DropOutNeuralNet"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.scalanlp" % "breeze_2.11" % "0.11.2",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

lazy val DropOutNeuralNet_dev = (project in file(".")).
  settings(
    name := "DropOutNeuralNetDevBuild"
  ).dependsOn(CoreNLP,nnutil,util)

lazy val CoreNLP = RootProject ( file("../CoreNLP") )
lazy val nnutil = RootProject ( file("../NeuralNetworkUtility") )
lazy val util = RootProject ( file("../Utility"))




