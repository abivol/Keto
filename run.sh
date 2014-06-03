
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

java -Xmx12g -cp ./target/classes:`cat classpath.out`  bme.App
