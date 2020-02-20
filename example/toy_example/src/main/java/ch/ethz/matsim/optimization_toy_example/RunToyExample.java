package ch.ethz.matsim.optimization_toy_example;

import java.util.Arrays;
import java.util.Random;

import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.NetworkFactory;
import org.matsim.api.core.v01.network.Node;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.PopulationFactory;
import org.matsim.core.config.CommandLine;
import org.matsim.core.config.CommandLine.ConfigurationException;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.PlanCalcScoreConfigGroup.ActivityParams;
import org.matsim.core.config.groups.PlanCalcScoreConfigGroup.ModeParams;
import org.matsim.core.config.groups.PlanCalcScoreConfigGroup.ScoringParameterSet;
import org.matsim.core.config.groups.PlansCalcRouteConfigGroup.ModeRoutingParams;
import org.matsim.core.config.groups.StrategyConfigGroup;
import org.matsim.core.config.groups.StrategyConfigGroup.StrategySettings;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.scenario.ScenarioUtils;

public class RunToyExample {
	static public void main(String[] args) throws ConfigurationException {
		Config config = ConfigUtils.createConfig();

		// CONFIG PART

		// Some initial setup
		config.controler().setOverwriteFileSetting(OverwriteFileSetting.deleteDirectoryIfExists);
		config.controler().setOutputDirectory("simulation_output");
		config.controler().setWritePlansInterval(1);

		// Setting up the activity scoring parameters ...
		ScoringParameterSet scoringParameters = config.planCalcScore().getOrCreateScoringParameters(null);

		// ... for home
		ActivityParams homeParams = scoringParameters.getOrCreateActivityParams("home");
		homeParams.setTypicalDuration(1.0);
		homeParams.setScoringThisActivityAtAll(false);

		// ... for work
		ActivityParams workParams = scoringParameters.getOrCreateActivityParams("work");
		workParams.setTypicalDuration(1.0);
		workParams.setScoringThisActivityAtAll(false);

		// ... for car
		ModeParams carParams = scoringParameters.getOrCreateModeParams("car");
		carParams.setConstant(0.0);
		carParams.setMarginalUtilityOfTraveling(-0.1);
		carParams.setMonetaryDistanceRate(0.0);

		// ... for pt
		ModeParams ptParams = scoringParameters.getOrCreateModeParams("pt");
		ptParams.setConstant(0.0);
		ptParams.setMarginalUtilityOfTraveling(-0.2);
		ptParams.setMonetaryDistanceRate(0.0);

		// Setting up replanning ...
		StrategyConfigGroup strategyConfig = config.strategy();
		strategyConfig.clearStrategySettings();
		strategyConfig.setMaxAgentPlanMemorySize(3);

		// ... for selection
		StrategySettings selectionStrategy = new StrategySettings();
		selectionStrategy.setStrategyName("ChangeExpBeta");
		selectionStrategy.setWeight(0.9);
		strategyConfig.addStrategySettings(selectionStrategy);

		// ... for innovation
		StrategySettings innovationStrategy = new StrategySettings();
		innovationStrategy.setStrategyName("ChangeTripMode");
		innovationStrategy.setWeight(0.1);
		strategyConfig.addStrategySettings(innovationStrategy);

		config.changeMode().setModes(new String[] { "car", "pt" });

		// Setting up routing for public transport
		ModeRoutingParams ptRoutingParams = config.plansCalcRoute().getOrCreateModeRoutingParams("pt");
		ptRoutingParams.setTeleportedModeFreespeedFactor(null);
		ptRoutingParams.setTeleportedModeSpeed(8.33);
		ptRoutingParams.setBeelineDistanceFactor(1.0);

		CommandLine cmd = new CommandLine.Builder(args).build();
		cmd.applyConfiguration(config);

		Scenario scenario = ScenarioUtils.createScenario(config);

		Network network = scenario.getNetwork();
		NetworkFactory networkFactory = network.getFactory();

		// N1 ------ N2 ------ N3 ------ N4 ------ N5
		// L1 L2 L3 L4

		Node node1 = networkFactory.createNode(Id.createNodeId("N1"), new Coord(0.0, 1000.0));
		Node node2 = networkFactory.createNode(Id.createNodeId("N2"), new Coord(0.0, 2000.0));
		Node node3 = networkFactory.createNode(Id.createNodeId("N3"), new Coord(0.0, 3000.0));
		Node node4 = networkFactory.createNode(Id.createNodeId("N4"), new Coord(0.0, 4000.0));
		Node node5 = networkFactory.createNode(Id.createNodeId("N5"), new Coord(0.0, 5000.0));

		Link link1 = networkFactory.createLink(Id.createLinkId("L1"), node1, node2);
		Link link2 = networkFactory.createLink(Id.createLinkId("L2"), node2, node3);
		Link link3 = networkFactory.createLink(Id.createLinkId("L3"), node3, node4);
		Link link4 = networkFactory.createLink(Id.createLinkId("L4"), node4, node5);

		network.addNode(node1);
		network.addNode(node2);
		network.addNode(node3);
		network.addNode(node4);
		network.addNode(node5);

		network.addLink(link1);
		network.addLink(link2);
		network.addLink(link3);
		network.addLink(link4);

		for (Link link : Arrays.asList(link1, link2, link3, link4)) {
			link.setCapacity(700);
			link.setFreespeed(8.33);
		}

		Population population = scenario.getPopulation();
		PopulationFactory factory = population.getFactory();

		Random random = new Random(0);

		for (int k = 0; k < 1000; k++) {
			Person person = factory.createPerson(Id.createPersonId(k));
			population.addPerson(person);

			Plan plan = factory.createPlan();
			person.addPlan(plan);

			Activity startActivity = factory.createActivityFromLinkId("home", Id.createLinkId("L1"));
			startActivity.setEndTime(random.nextDouble() * 3600.0);
			startActivity.setCoord(node2.getCoord());
			plan.addActivity(startActivity);

			Leg leg = factory.createLeg("car");
			plan.addLeg(leg);

			Activity endActivity = factory.createActivityFromLinkId("work", Id.createLinkId("L4"));
			endActivity.setCoord(node4.getCoord());
			plan.addActivity(endActivity);
		}

		Controler controller = new Controler(scenario);
		controller.run();
	}
}
