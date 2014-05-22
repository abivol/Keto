package bme

import java.util.Set;
import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.util.FoldUtils;
import edu.umd.cs.bachuai13.util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries
import edu.ucsc.cs.utils.Evaluator;

import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator



//dataSet = "fourforums"
dataSet = "PRAD"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

PSLModel model = new PSLModel(this, data)

/*
 * Label predicates
 */

// PatientLabel(P) -- observed var.
model.add predicate: "PatientLabel" , types:[ArgumentType.UniqueID]
// GeneLabel(G) -- latent var.
model.add predicate: "GeneLabel" , types:[ArgumentType.UniqueID]
// TargetPatientLabel(P) -- target var.
model.add predicate: "TargetPatientLabel" , types:[ArgumentType.UniqueID]

/*
 * Evidence (observed) predicates
 */
model.add predicate: "ExpUp" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "ExpDown" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "MutPlus" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]  // GOF
model.add predicate: "MutMinus" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]  // LOF

// Active(G, P) -- latent
model.add predicate: "Active" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// Activates(G1, G2) -- observed
model.add predicate: "Activates" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// Inhibits(G1, G2) -- observed
model.add predicate: "Inhibits" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// Similar(G1, G2) -- observed
model.add predicate: "Similar" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// mutual exclusivity - no need to enforce these, they are all observed
// model.add rule : ExpUp(G, P) + ExpDown(G, P) <= 1.0
// model.add rule : MutPlus(G, P) + MutMinus(G, P) <= 1.0

// rules to infer gene activity from evidence
model.add rule : ExpUp(G, P) >> Active(G, P), weight : 1
model.add rule : ExpDown(G, P) >> ~Active(G, P), weight : 1
model.add rule : MutPlus(G,P) >> ~Active(G, P), weight : 1
model.add rule : MutMinus(G, P) >> Active(G, P), weight : 1

// rules to relate gene activities via gene interaction network
model.add rule : (Inhibits(G1, G2) & Active(G1, P)) >> ~Active(G2, P), weight : 1
model.add rule : (Activates(G1, G2) & Active(G1, P)) >> Active(G2, P), weight : 1
model.add rule : (Similar(G1, G2) & GeneLabel(G1)) >> GeneLabel(G2), weight : 1

// label inference
model.add rule : (PatientLabel(P) & Active(G, P)) >> GeneLabel(G), weight : 1
model.add rule : (GeneLabel(G) & Active(G, P)) >> TargetPatientLabel(P), weight : 1

/////////////////////////////////////////////////////////////////////

// add rules for complexes
model.add predicate: "Complex_3" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (Complex_3(G1, G2, G3, G_OUT) & Active(G1, P) & Active(G2, P) & Active(G3, P)) >> Active(G_OUT, P), weight : 1


/*
 * Inserting data into the data store
 */
//fold = 1

//foldStr = "fold" + String.valueOf(fold) + java.io.File.separator;

/* training partitions */
Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition dummy_tr = new Partition(3);

/*testing partitions */
Partition observed_te = new Partition(4);
Partition predict_te = new Partition(5);
Partition dummy_te = new Partition(6);

/*separate partitions for the gold standard truth for testing */
Partition PatientLabelTruth = new Partition(7);


//def dir = 'data'+java.io.File.separator+ foldStr + 'train'+java.io.File.separator;
def traindir = 'data' + java.io.File.separator + 'train' + java.io.File.separator;

/*
 * Observed data partition
 * 
 */

inserter = data.getInserter(ExpUp, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "ExpUp.csv", ",");

inserter = data.getInserter(ExpDown, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "ExpDown.csv", ",");

inserter = data.getInserter(MutPlus, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "MutPlus.csv", ",");

inserter = data.getInserter(MutMinus, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "MutMinus.csv", ",");

inserter = data.getInserter(PatientLabel, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "PatientLabel.csv", ",");

/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(TargetPatientLabel, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "TargetPatientLabel.csv", ",")


/*
 * Used later on to populate training DB with all possible interactions
 * ??? predicates on latents ???
 */

inserter = data.getInserter(Activates, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Activates.csv", ",")

inserter = data.getInserter(Inhibits, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Inhibits.csv", ",")

inserter = data.getInserter(Similar, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Similar.csv", ",")

inserter = data.getInserter(TargetPatientLabel, dummy_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "BogusPatientLabel.csv", ",")

/*
 * Testing split for model inference
 * Observed partitions
 */

//def testdir = 'data'+java.io.File.separator+ foldStr + 'test'+java.io.File.separator;
def testdir = 'data' + java.io.File.separator + 'test' + java.io.File.separator;

inserter = data.getInserter(ExpUp, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "ExpUp.csv", ",");

inserter = data.getInserter(ExpDown, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "ExpDown.csv", ",");

inserter = data.getInserter(MutPlus, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "MutPlus.csv", ",");

inserter = data.getInserter(MutMinus, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "MutMinus.csv", ",");

/*
 * Random variable partitions
 */

inserter = data.getInserter(TargetPatientLabel, PatientLabelTruth)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "PatientLabel.csv", ",");

/* ?? predicates on latents ?? */

inserter = data.getInserter(Activates, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Activates.csv", ",")

inserter = data.getInserter(Inhibits, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Inhibits.csv", ",")

inserter = data.getInserter(Similar, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Similar.csv", ",")

/*to populate testDB with the correct rvs
 */

inserter = data.getInserter(TargetPatientLabel, dummy_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "BogusPatientLabel.csv", ",");


/*
 * Set up training databases for weight learning using training set
 */

Database distributionDB = data.getDatabase(predict_tr, [ExpUp, ExpDown, MutPlus, MutMinus] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [TargetPatientLabel] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [Activates, Inhibits, Similar, TargetPatientLabel] as Set)

/* Populate distribution DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, Activates);
dbPop.populateFromDB(dummy_DB, Inhibits);
dbPop.populateFromDB(dummy_DB, Similar);
dbPop.populateFromDB(dummy_DB, TargetPatientLabel);


/*
HardEM weightLearning = new HardEM(model, distributionDB, truthDB, cb);
println "about to start weight learning"
weightLearning.learn();
println " finished weight learning "
weightLearning.close();
*/
/*
 MaxPseudoLikelihood mple = new MaxPseudoLikelihood(model, trainDB, truthDB, cb);
 println "about to start weight learning"
 mple.learn();
 println " finished weight learning "
 mlpe.close();
 */

println model;


distributionDB.close()
truthDB.close()
dummy_DB.close()
