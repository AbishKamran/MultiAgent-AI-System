#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Multi-Agent AI System
Tests goal satisfaction, agent chaining, robustness, and performance
"""

import asyncio
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
from unittest.mock import patch, AsyncMock, Mock

from multi_agent_system import (
    MultiAgentOrchestrator,
    AgentStatus,
    AgentResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Standard evaluation result format"""
    test_name: str
    score: float
    max_score: float
    success: bool
    details: Dict[str, Any]
    execution_time: float
    timestamp: str

class MultiAgentEvaluator:
    """Main evaluation class for the multi-agent system"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.orchestrator = MultiAgentOrchestrator(api_keys)
        self.evaluation_results = []
        
        # Test scenarios
        self.test_scenarios = {
            "spacex_weather": {
                "goal": "Find the next SpaceX launch, check weather at that location, then summarize if it may be delayed",
                "expected_agents": ["spacex_agent", "weather_agent", "analyzer_agent"],
                "expected_outputs": ["launch", "weather", "delay_probability", "risk_level"],
                "data_dependencies": ["location", "weather_data", "analysis"]
            },
            "crypto_sentiment": {
                "goal": "Get Bitcoin price and recent news, then analyze market sentiment",
                "expected_agents": ["crypto_agent", "news_agent", "sentiment_agent"],
                "expected_outputs": ["prices", "articles", "sentiment", "market_impact"],
                "data_dependencies": ["crypto_data", "news_data", "sentiment_analysis"]
            },
            "ethereum_focus": {
                "goal": "Analyze Ethereum market trends with news sentiment",
                "expected_agents": ["crypto_agent", "news_agent", "sentiment_agent"],
                "expected_outputs": ["ethereum", "news", "sentiment"],
                "data_dependencies": ["eth_price", "eth_news", "sentiment_score"]
            }
        }
    
    async def run_all_evaluations(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        logger.info("Starting comprehensive multi-agent system evaluation")
        start_time = time.time()
        
        # Evaluation 1: Goal Satisfaction
        goal_satisfaction_results = await self.evaluate_goal_satisfaction()
        
        # Evaluation 2: Agent Chaining and Data Flow
        chaining_results = await self.evaluate_agent_chaining()
        
        # Additional evaluations
        robustness_results = await self.evaluate_robustness()
        performance_results = await self.evaluate_performance()
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            "evaluation_summary": {
                "total_tests": len(self.evaluation_results),
                "total_time": round(total_time, 2),
                "timestamp": datetime.now().isoformat()
            },
            "goal_satisfaction": goal_satisfaction_results,
            "agent_chaining": chaining_results,
            "robustness": robustness_results,
            "performance": performance_results,
            "detailed_results": self.evaluation_results,
            "overall_score": self._calculate_overall_score()
        }
        
        self._save_results(final_results)
        return final_results
    
    async def evaluate_goal_satisfaction(self) -> Dict[str, Any]:
        """Evaluation 1: Test how well the system satisfies different types of goals"""
        logger.info("Running Goal Satisfaction Evaluation")
        results = {}
        
        for scenario_name, scenario in self.test_scenarios.items():
            start_time = time.time()
            
            try:
                # Execute the goal with mocked API responses
                with self._mock_api_responses(scenario_name):
                    execution_result = await self.orchestrator.execute_goal(scenario["goal"])
                
                # Evaluate satisfaction
                satisfaction_score = self._evaluate_goal_components(
                    execution_result, 
                    scenario["expected_outputs"]
                )
                
                agent_coverage_score = self._evaluate_agent_coverage(
                    execution_result,
                    scenario["expected_agents"]
                )
                
                data_flow_score = self._evaluate_data_dependencies(
                    execution_result,
                    scenario["data_dependencies"]
                )
                
                # Combined score
                total_score = (satisfaction_score + agent_coverage_score + data_flow_score) / 3
                
                execution_time = time.time() - start_time
                
                eval_result = EvaluationResult(
                    test_name=f"goal_satisfaction_{scenario_name}",
                    score=total_score,
                    max_score=1.0,
                    success=execution_result.get("success", False),
                    details={
                        "goal": scenario["goal"],
                        "satisfaction_score": satisfaction_score,
                        "agent_coverage_score": agent_coverage_score,
                        "data_flow_score": data_flow_score,
                        "agents_used": list(execution_result.get("results", {}).keys()),
                        "iterations": execution_result.get("iterations", 0),
                        "final_output_keys": list(execution_result.get("final_output", {}).keys())
                    },
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
                results[scenario_name] = eval_result
                self.evaluation_results.append(eval_result)
                
                logger.info(f"Goal satisfaction for {scenario_name}: {total_score:.2%}")
                
            except Exception as e:
                logger.error(f"Goal satisfaction test failed for {scenario_name}: {str(e)}")
                results[scenario_name] = EvaluationResult(
                    test_name=f"goal_satisfaction_{scenario_name}",
                    score=0.0,
                    max_score=1.0,
                    success=False,
                    details={"error": str(e)},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
        
        return results
    
    async def evaluate_agent_chaining(self) -> Dict[str, Any]:
        """Evaluation 2: Test agent chaining and data flow quality"""
        logger.info("Running Agent Chaining Evaluation")
        results = {}
        
        # Test proper data passing between agents
        chaining_tests = [
            {
                "name": "spacex_to_weather_data_flow",
                "goal": "Find SpaceX launch and check weather",
                "source_agent": "spacex_agent",
                "target_agent": "weather_agent",
                "required_data": ["latitude", "longitude"]
            },
            {
                "name": "crypto_to_news_data_flow", 
                "goal": "Get crypto prices and related news",
                "source_agent": "crypto_agent",
                "target_agent": "news_agent",
                "required_data": ["analyzed_coins"]
            },
            {
                "name": "news_to_sentiment_data_flow",
                "goal": "Analyze crypto news sentiment",
                "source_agent": "news_agent", 
                "target_agent": "sentiment_agent",
                "required_data": ["articles"]
            }
        ]
        
        for test in chaining_tests:
            start_time = time.time()
            
            try:
                with self._mock_api_responses("spacex_weather"):
                    execution_result = await self.orchestrator.execute_goal(test["goal"])
                
                # Check if data flowed correctly between agents
                chaining_score = self._evaluate_data_chaining(
                    execution_result,
                    test["source_agent"],
                    test["target_agent"],
                    test["required_data"]
                )
                
                execution_time = time.time() - start_time
                
                eval_result = EvaluationResult(
                    test_name=f"agent_chaining_{test['name']}",
                    score=chaining_score,
                    max_score=1.0,
                    success=chaining_score > 0.5,
                    details={
                        "source_agent": test["source_agent"],
                        "target_agent": test["target_agent"],
                        "required_data": test["required_data"],
                        "data_flow_quality": chaining_score
                    },
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
                results[test["name"]] = eval_result
                self.evaluation_results.append(eval_result)
                
                logger.info(f"Agent chaining for {test['name']}: {chaining_score:.2%}")
                
            except Exception as e:
                logger.error(f"Agent chaining test failed for {test['name']}: {str(e)}")
        
        return results
    
    async def evaluate_robustness(self) -> Dict[str, Any]:
        """Test system robustness under various failure conditions"""
        logger.info("Running Robustness Evaluation")
        results = {}
        
        robustness_tests = [
            {
                "name": "api_failure_recovery",
                "description": "Test recovery when API calls fail",
                "failure_type": "api_error"
            },
            {
                "name": "partial_data_handling",
                "description": "Test handling of incomplete data",
                "failure_type": "partial_data"
            },
            {
                "name": "agent_sequence_interruption", 
                "description": "Test recovery when agent sequence is interrupted",
                "failure_type": "agent_failure"
            }
        ]
        
        for test in robustness_tests:
            start_time = time.time()
            
            try:
                # Simulate different failure conditions
                with self._mock_failure_scenarios(test["failure_type"]):
                    execution_result = await self.orchestrator.execute_goal(
                        "Test robustness with SpaceX weather goal"
                    )
                
                # Evaluate recovery
                recovery_score = self._evaluate_failure_recovery(execution_result)
                
                execution_time = time.time() - start_time
                
                eval_result = EvaluationResult(
                    test_name=f"robustness_{test['name']}",
                    score=recovery_score,
                    max_score=1.0,
                    success=recovery_score > 0.3,  # Lower threshold for robustness
                    details={
                        "failure_type": test["failure_type"],
                        "description": test["description"],
                        "iterations": execution_result.get("iterations", 0),
                        "final_success": execution_result.get("success", False)
                    },
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
                results[test["name"]] = eval_result
                self.evaluation_results.append(eval_result)
                
                logger.info(f"Robustness test {test['name']}: {recovery_score:.2%}")
                
            except Exception as e:
                logger.error(f"Robustness test failed for {test['name']}: {str(e)}")
        
        return results
    
    async def evaluate_performance(self) -> Dict[str, Any]:
        """Test system performance and efficiency"""
        logger.info("Running Performance Evaluation")
        results = {}
        
        # Performance metrics
        execution_times = []
        iteration_counts = []
        success_rates = []
        
        # Run multiple executions for statistical analysis
        for i in range(5):
            start_time = time.time()
            
            try:
                with self._mock_api_responses("spacex_weather"):
                    execution_result = await self.orchestrator.execute_goal(
                        "Performance test: SpaceX launch weather analysis"
                    )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                iteration_counts.append(execution_result.get("iterations", 0))
                success_rates.append(1.0 if execution_result.get("success", False) else 0.0)
                
            except Exception as e:
                logger.error(f"Performance test iteration {i} failed: {str(e)}")
                execution_times.append(60.0)  # Max timeout
                iteration_counts.append(3)    # Max iterations
                success_rates.append(0.0)
        
        # Calculate performance metrics
        avg_execution_time = statistics.mean(execution_times)
        avg_iterations = statistics.mean(iteration_counts)
        success_rate = statistics.mean(success_rates)
        
        # Performance score (lower time and iterations = better)
        time_score = max(0, 1 - (avg_execution_time / 30))  # 30s baseline
        iteration_score = max(0, 1 - (avg_iterations / 3))  # 3 iterations max
        performance_score = (time_score + iteration_score + success_rate) / 3
        
        eval_result = EvaluationResult(
            test_name="performance_analysis",
            score=performance_score,
            max_score=1.0,
            success=success_rate > 0.8,
            details={
                "avg_execution_time": round(avg_execution_time, 2),
                "avg_iterations": round(avg_iterations, 2),
                "success_rate": round(success_rate, 2),
                "time_score": round(time_score, 2),
                "iteration_score": round(iteration_score, 2),
                "sample_size": len(execution_times)
            },
            execution_time=sum(execution_times),
            timestamp=datetime.now().isoformat()
        )
        
        results["performance_analysis"] = eval_result
        self.evaluation_results.append(eval_result)
        
        logger.info(f"Performance score: {performance_score:.2%}")
        
        return results
    
    def _evaluate_goal_components(self, execution_result: Dict[str, Any], 
                                expected_outputs: List[str]) -> float:
        """Evaluate if expected output components are present"""
        final_output = execution_result.get("final_output", {})
        final_output_str = json.dumps(final_output).lower()
        
        found_components = 0
        for component in expected_outputs:
            if component.lower() in final_output_str:
                found_components += 1
        
        return found_components / len(expected_outputs) if expected_outputs else 0.0
    
    def _evaluate_agent_coverage(self, execution_result: Dict[str, Any],
                               expected_agents: List[str]) -> float:
        """Evaluate if expected agents were executed"""
        executed_agents = set(execution_result.get("results", {}).keys())
        expected_agents_set = set(expected_agents)
        
        coverage = len(executed_agents & expected_agents_set) / len(expected_agents_set)
        return coverage
    
    def _evaluate_data_dependencies(self, execution_result: Dict[str, Any],
                                  data_dependencies: List[str]) -> float:
        """Evaluate if data dependencies are satisfied"""
        results = execution_result.get("results", {})
        
        dependency_score = 0
        for dependency in data_dependencies:
            # Check if dependency data exists in any agent's output
            for agent_result in results.values():
                if isinstance(agent_result, dict):
                    agent_data = agent_result.get("data", {})
                    if dependency.lower() in json.dumps(agent_data).lower():
                        dependency_score += 1
                        break
        
        return dependency_score / len(data_dependencies) if data_dependencies else 0.0
    
    def _evaluate_data_chaining(self, execution_result: Dict[str, Any],
                              source_agent: str, target_agent: str,
                              required_data: List[str]) -> float:
        """Evaluate quality of data flow between specific agents"""
        results = execution_result.get("results", {})
        
        if source_agent not in results or target_agent not in results:
            return 0.0
        
        source_data = results[source_agent].get("data", {})
        
        # Check if required data from source is available
        chaining_score = 0
        for data_key in required_data:
            if self._find_nested_key(source_data, data_key):
                chaining_score += 1
        
        return chaining_score / len(required_data) if required_data else 0.0
    
    def _find_nested_key(self, data: dict, key: str) -> bool:
        """Recursively find a key in nested dictionary"""
        if key in data:
            return True
        
        for value in data.values():
            if isinstance(value, dict):
                if self._find_nested_key(value, key):
                    return True
        
        return False
    
    def _evaluate_failure_recovery(self, execution_result: Dict[str, Any]) -> float:
        """Evaluate how well system recovered from failures"""
        iterations = execution_result.get("iterations", 1)
        success = execution_result.get("success", False)
        
        if success:
            # Good recovery if succeeded, better if fewer iterations needed
            return max(0.5, 1.0 - (iterations - 1) * 0.2)
        else:
            # Partial credit for attempting recovery
            return min(0.3, iterations * 0.1)
    
    def _calculate_overall_score(self) -> Dict[str, float]:
        """Calculate overall system evaluation score"""
        if not self.evaluation_results:
            return {"overall_score": 0.0}
        
        # Group scores by evaluation type
        goal_scores = [r.score for r in self.evaluation_results if "goal_satisfaction" in r.test_name]
        chaining_scores = [r.score for r in self.evaluation_results if "agent_chaining" in r.test_name]
        robustness_scores = [r.score for r in self.evaluation_results if "robustness" in r.test_name]
        performance_scores = [r.score for r in self.evaluation_results if "performance" in r.test_name]
        
        return {
            "overall_score": statistics.mean([r.score for r in self.evaluation_results]),
            "goal_satisfaction_avg": statistics.mean(goal_scores) if goal_scores else 0.0,
            "agent_chaining_avg": statistics.mean(chaining_scores) if chaining_scores else 0.0,
            "robustness_avg": statistics.mean(robustness_scores) if robustness_scores else 0.0,
            "performance_avg": statistics.mean(performance_scores) if performance_scores else 0.0,
            "total_tests": len(self.evaluation_results),
            "success_rate": len([r for r in self.evaluation_results if r.success]) / len(self.evaluation_results)
        }
    
    def _mock_api_responses(self, scenario: str):
        """Context manager for mocking API responses"""
        return patch('aiohttp.ClientSession.get', side_effect=self._get_mock_response_handler(scenario))
    
    def _mock_failure_scenarios(self, failure_type: str):
        """Context manager for mocking various failure scenarios"""
        if failure_type == "api_error":
            return patch('aiohttp.ClientSession.get', side_effect=Exception("API Error"))
        elif failure_type == "partial_data":
            return patch('aiohttp.ClientSession.get', side_effect=self._get_partial_data_handler())
        elif failure_type == "agent_failure":
            return patch.object(self.orchestrator.agents["weather_agent"], 'execute',
                              side_effect=Exception("Agent failure"))
    
    def _get_mock_response_handler(self, scenario: str):
        """Get appropriate mock response handler for scenario"""
        async def mock_response_handler(*args, **kwargs):
            mock_response = AsyncMock()
            
            # Determine response based on URL
            url = str(args[0]) if args else ""
            
            if "spacex" in url:
                if "launches" in url:
                    mock_response.json.return_value = [{
                        "name": "Starlink-6-1",
                        "date_utc": "2024-02-15T10:30:00.000Z",
                        "date_local": "2024-02-15T05:30:00-05:00",
                        "flight_number": 100,
                        "launchpad": "5e9e4502f509094188566f88"
                    }]
                else:  # launchpad
                    mock_response.json.return_value = {
                        "full_name": "Kennedy Space Center LC-39A",
                        "locality": "Merritt Island",
                        "region": "Florida",
                        "latitude": 28.6080585,
                        "longitude": -80.6039558
                    }
            elif "openweathermap" in url:
                if "weather" in url:
                    mock_response.json.return_value = {
                        "main": {"temp": 22.5, "humidity": 65, "pressure": 1013},
                        "wind": {"speed": 8.5, "deg": 180},
                        "weather": [{"description": "clear sky"}],
                        "visibility": 10000
                    }
                else:  # forecast
                    mock_response.json.return_value = {
                        "list": [{
                            "dt_txt": "2024-02-15 12:00:00",
                            "main": {"temp": 24.0},
                            "wind": {"speed": 10.0},
                            "weather": [{"description": "partly cloudy"}],
                            "rain": {}, "snow": {}
                        }]
                    }
            elif "coingecko" in url:
                mock_response.json.return_value = {
                    "bitcoin": {"usd": 45000, "usd_24h_change": 2.5}
                }
            elif "newsapi" in url:
                mock_response.json.return_value = {
                    "articles": [{
                        "title": "Bitcoin rises on positive sentiment",
                        "description": "Market shows bullish trends",
                        "url": "https://example.com",
                        "publishedAt": "2024-02-15T12:00:00Z",
                        "source": {"name": "CryptoNews"}
                    }]
                }
            
            return mock_response
        
        return mock_response_handler
    
    def _get_partial_data_handler(self):
        """Handler for partial/incomplete data scenarios"""
        async def partial_response_handler(*args, **kwargs):
            mock_response = AsyncMock()
            # Return incomplete data to test robustness
            mock_response.json.return_value = {"incomplete": "data"}
            return mock_response
        
        return partial_response_handler
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_agent_evaluation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filename}")

async def main():
    """Main function to run evaluations"""
    # Mock API keys for testing
    api_keys = {
        "OPENWEATHER_API_KEY": "test_weather_key",
        "NEWS_API_KEY": "test_news_key"
    }
    
    evaluator = MultiAgentEvaluator(api_keys)
    
    print("=" * 60)
    print("MULTI-AGENT AI SYSTEM EVALUATION SUITE")
    print("=" * 60)
    
    results = await evaluator.run_all_evaluations()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    overall_scores = results["overall_score"]
    print(f"Overall Score: {overall_scores['overall_score']:.2%}")
    print(f"Goal Satisfaction: {overall_scores['goal_satisfaction_avg']:.2%}")
    print(f"Agent Chaining: {overall_scores['agent_chaining_avg']:.2%}")
    print(f"Robustness: {overall_scores['robustness_avg']:.2%}")
    print(f"Performance: {overall_scores['performance_avg']:.2%}")
    print(f"Success Rate: {overall_scores['success_rate']:.2%}")
    print(f"Total Tests: {overall_scores['total_tests']}")
    
    # Detailed breakdown
    print("\n" + "-" * 40)
    print("DETAILED RESULTS")
    print("-" * 40)
    
    for category, category_results in results.items():
        if category in ["goal_satisfaction", "agent_chaining", "robustness", "performance"]:
            print(f"\n{category.upper()}:")
            for test_name, test_result in category_results.items():
                if hasattr(test_result, 'score'):
                    status = "✓" if test_result.success else "✗"
                    print(f"  {status} {test_name}: {test_result.score:.2%}")

if __name__ == "__main__":
    asyncio.run(main())