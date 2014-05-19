
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

java -cp ./target/classes:`cat classpath.out`  bme.App
