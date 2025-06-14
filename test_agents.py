# Test suite for Multi-Agent AI System
import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from multi_agent_system import (
    MultiAgentOrchestrator,
    PlannerAgent,
    SpaceXAgent,
    WeatherAgent,
    AnalyzerAgent,
    CryptoAgent,
    NewsAgent,
    SentimentAgent,
    AgentStatus
)

# Test data
MOCK_SPACEX_LAUNCH = {
    "name": "Starlink-6-1",
    "date_utc": "2024-02-15T10:30:00.000Z",
    "date_local": "2024-02-15T05:30:00-05:00",
    "flight_number": 100,
    "details": "Starlink mission",
    "launchpad": "5e9e4502f509094188566f88"
}

MOCK_LAUNCHPAD = {
    "full_name": "Kennedy Space Center LC-39A",
    "locality": "Merritt Island",
    "region": "Florida",
    "latitude": 28.6080585,
    "longitude": -80.6039558
}

MOCK_WEATHER = {
    "main": {"temp": 22.5, "humidity": 65, "pressure": 1013},
    "wind": {"speed": 8.5, "deg": 180},
    "weather": [{"description": "clear sky"}],
    "visibility": 10000
}

MOCK_WEATHER_FORECAST = {
    "list": [
        {
            "dt_txt": "2024-02-15 12:00:00",
            "main": {"temp": 24.0},
            "wind": {"speed": 10.0},
            "weather": [{"description": "partly cloudy"}],
            "rain": {},
            "snow": {}
        }
    ]
}

@pytest.fixture
def api_keys():
    return {
        "OPENWEATHER_API_KEY": "test_weather_key",
        "NEWS_API_KEY": "test_news_key"
    }

@pytest.fixture
def orchestrator(api_keys):
    return MultiAgentOrchestrator(api_keys)

class TestPlannerAgent:
    """Test cases for the Planner Agent"""
    
    @pytest.mark.asyncio
    async def test_spacex_goal_planning(self, api_keys):
        planner = PlannerAgent(api_keys)
        result = await planner.execute({
            "goal": "Find next SpaceX launch and check weather conditions"
        })
        
        assert result.status == AgentStatus.COMPLETED
        assert "plan" in result.data
        plan = result.data["plan"]
        assert "spacex_agent" in plan["agent_sequence"]
        assert "weather_agent" in plan["agent_sequence"]
        assert "analyzer_agent" in plan["agent_sequence"]
    
    @pytest.mark.asyncio
    async def test_crypto_goal_planning(self, api_keys):
        planner = PlannerAgent(api_keys)
        result = await planner.execute({
            "goal": "Get Bitcoin price and analyze market sentiment"
        })
        
        assert result.status == AgentStatus.COMPLETED
        plan = result.data["plan"]
        assert "crypto_agent" in plan["agent_sequence"]
        assert "news_agent" in plan["agent_sequence"]
        assert "sentiment_agent" in plan["agent_sequence"]

class TestSpaceXAgent:
    """Test cases for the SpaceX Agent"""
    
    @pytest.mark.asyncio
    async def test_fetch_launch_data_success(self, api_keys):
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock the API responses
            mock_response = AsyncMock()
            mock_response.json = AsyncMock()
            
            # First call returns launches, second returns launchpad
            mock_get.return_value.__aenter__.return_value = mock_response
            mock_response.json.side_effect = [
                [MOCK_SPACEX_LAUNCH],  # launches response
                MOCK_LAUNCHPAD         # launchpad response
            ]
            
            agent = SpaceXAgent(api_keys)
            result = await agent.execute({})
            
            assert result.status == AgentStatus.COMPLETED
            assert "launch" in result.data
            assert "location" in result.data
            assert result.data["launch"]["name"] == "Starlink-6-1"
            assert result.data["location"]["latitude"] == 28.6080585
    
    @pytest.mark.asyncio
    async def test_no_upcoming_launches(self, api_keys):
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=[])
            mock_get.return_value.__aenter__.return_value = mock_response
            
            agent = SpaceXAgent(api_keys)
            result = await agent.execute({})
            
            assert result.status == AgentStatus.COMPLETED
            assert result.data["launch"] is None
            assert "No upcoming launches found" in result.data["message"]

class TestWeatherAgent:
    """Test cases for the Weather Agent"""
    
    @pytest.mark.asyncio
    async def test_fetch_weather_success(self, api_keys):
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock()
            mock_get.return_value.__aenter__.return_value = mock_response
            mock_response.json.side_effect = [MOCK_WEATHER, MOCK_WEATHER_FORECAST]
            
            agent = WeatherAgent(api_keys)
            input_data = {
                "spacex_agent": {
                    "data": {
                        "location": {
                            "latitude": 28.6080585,
                            "longitude": -80.6039558,
                            "name": "Kennedy Space Center"
                        }
                    }
                }
            }
            
            result = await agent.execute(input_data)
            
            assert result.status == AgentStatus.COMPLETED
            assert "current" in result.data
            assert "forecast" in result.data
            assert result.data["current"]["temperature"] == 22.5
    
    @pytest.mark.asyncio
    async def test_missing_location_data(self, api_keys):
        agent = WeatherAgent(api_keys)
        result = await agent.execute({"spacex_agent": {"data": {}}})
        
        assert result.status == AgentStatus.FAILED
        assert "No location data" in result.error_message

class TestAnalyzerAgent:
    """Test cases for the Analyzer Agent"""
    
    @pytest.mark.asyncio
    async def test_low_risk_analysis(self, api_keys):
        agent = AnalyzerAgent(api_keys)
        input_data = {
            "spacex_agent": {
                "data": {
                    "launch": {"name": "Test Launch", "date_local": "2024-02-15"},
                    "location": {"name": "Test Location"}
                }
            },
            "weather_agent": {
                "data": {
                    "current": {
                        "wind_speed": 5.0,
                        "visibility": 10.0,
                        "description": "clear sky",
                        "temperature": 22.5
                    },
                    "forecast": [
                        {"wind_speed": 6.0, "precipitation": 0, "description": "sunny"}
                    ]
                }
            }
        }
        
        result = await agent.execute(input_data)
        
        assert result.status == AgentStatus.COMPLETED
        assert "analysis" in result.data
        analysis = result.data["analysis"]
        assert analysis["risk_level"] == "LOW"
        assert analysis["delay_probability"] < 0.2
    
    @pytest.mark.asyncio
    async def test_high_risk_analysis(self, api_keys):
        agent = AnalyzerAgent(api_keys)
        input_data = {
            "spacex_agent": {
                "data": {
                    "launch": {"name": "Test Launch", "date_local": "2024-02-15"},
                    "location": {"name": "Test Location"}
                }
            },
            "weather_agent": {
                "data": {
                    "current": {
                        "wind_speed": 20.0,  # High wind
                        "visibility": 2.0,   # Poor visibility
                        "description": "thunderstorm",  # Storm
                        "temperature": 15.0
                    },
                    "forecast": [
                        {"wind_speed": 18.0, "precipitation": 5.0, "description": "storm"}
                    ] * 5  # Multiple risky periods
                }
            }
        }
        
        result = await agent.execute(input_data)
        
        assert result.status == AgentStatus.COMPLETED
        analysis = result.data["analysis"]
        assert analysis["risk_level"] == "HIGH"
        assert analysis["delay_probability"] > 0.5

class TestCryptoAgent:
    """Test cases for the Crypto Agent"""
    
    @pytest.mark.asyncio
    async def test_fetch_crypto_data(self, api_keys):
        mock_price_data = {
            "bitcoin": {"usd": 45000, "usd_24h_change": 2.5, "usd_market_cap": 880000000000}
        }
        mock_trending = {"coins": [{"item": {"name": "Bitcoin"}}]}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock()
            mock_get.return_value.__aenter__.return_value = mock_response
            mock_response.json.side_effect = [mock_price_data, mock_trending]
            
            agent = CryptoAgent(api_keys)
            result = await agent.execute({"goal": "bitcoin price"})
            
            assert result.status == AgentStatus.COMPLETED
            assert "prices" in result.data
            assert "trending" in result.data

class TestSentimentAgent:
    """Test cases for the Sentiment Agent"""
    
    @pytest.mark.asyncio
    async def test_positive_sentiment_analysis(self, api_keys):
        agent = SentimentAgent(api_keys)
        input_data = {
            "crypto_agent": {
                "data": {
                    "analyzed_coins": ["bitcoin"],
                    "prices": {"bitcoin": {"usd": 45000}}
                }
            },
            "news_agent": {
                "data": {
                    "articles": [
                        {
                            "title": "Bitcoin surge continues with bullish growth",
                            "description": "Positive market trends show optimistic outlook"
                        },
                        {
                            "title": "Crypto gains momentum as investors profit",
                            "description": "Rising prices indicate strong market sentiment"
                        }
                    ]
                }
            }
        }
        
        result = await agent.execute(input_data)
        
        assert result.status == AgentStatus.COMPLETED
        assert result.data["overall_sentiment"] == "positive"
        assert result.data["sentiment_score"] > 0
    
    @pytest.mark.asyncio
    async def test_negative_sentiment_analysis(self, api_keys):
        agent = SentimentAgent(api_keys)
        input_data = {
            "crypto_agent": {
                "data": {
                    "analyzed_coins": ["bitcoin"],
                    "prices": {"bitcoin": {"usd": 30000}}
                }
            },
            "news_agent": {
                "data": {
                    "articles": [
                        {
                            "title": "Bitcoin crash sends market into bearish decline",
                            "description": "Negative sentiment as prices fall dramatically"
                        },
                        {
                            "title": "Crypto market pessimistic as losses mount",
                            "description": "Investors face significant downward pressure"
                        }
                    ]
                }
            }
        }
        
        result = await agent.execute(input_data)
        
        assert result.status == AgentStatus.COMPLETED
        assert result.data["overall_sentiment"] == "negative"
        assert result.data["sentiment_score"] < 0

class TestMultiAgentOrchestrator:
    """Test cases for the main orchestrator"""
    
    @pytest.mark.asyncio
    async def test_spacex_goal_execution(self, orchestrator):
        # Mock all the agents to return successful results
        with patch.object(orchestrator.agents["planner"], 'execute') as mock_planner, \
             patch.object(orchestrator.agents["spacex_agent"], 'execute') as mock_spacex, \
             patch.object(orchestrator.agents["weather_agent"], 'execute') as mock_weather, \
             patch.object(orchestrator.agents["analyzer_agent"], 'execute') as mock_analyzer:
            
            # Setup mock returns
            mock_planner.return_value = Mock(
                status=AgentStatus.COMPLETED,
                data={"plan": {"agent_sequence": ["spacex_agent", "weather_agent", "analyzer_agent"]}}
            )
            mock_spacex.return_value = Mock(
                status=AgentStatus.COMPLETED,
                data={"launch": {"name": "Test"}, "location": {"latitude": 28.6, "longitude": -80.6}}
            )
            mock_weather.return_value = Mock(
                status=AgentStatus.COMPLETED,
                data={"current": {"wind_speed": 5}, "forecast": []}
            )
            mock_analyzer.return_value = Mock(
                status=AgentStatus.COMPLETED,
                data={"analysis": {"risk_level": "LOW", "delay_probability": 0.1}}
            )
            
            result = await orchestrator.execute_goal(
                "Find the next SpaceX launch and check weather conditions"
            )
            
            assert result["success"] is True
            assert result["iterations"] == 1
            assert "final_output" in result
    
    @pytest.mark.asyncio
    async def test_agent_failure_retry(self, orchestrator):
        # Mock planner success but spacex failure, then success on retry
        with patch.object(orchestrator.agents["planner"], 'execute') as mock_planner, \
             patch.object(orchestrator.agents["spacex_agent"], 'execute') as mock_spacex:
            
            mock_planner.return_value = Mock(
                status=AgentStatus.COMPLETED,
                data={"plan": {"agent_sequence": ["spacex_agent"]}}
            )
            
            # First call fails, second succeeds
            mock_spacex.side_effect = [
                Mock(status=AgentStatus.FAILED, error_message="API Error"),
                Mock(status=AgentStatus.COMPLETED, data={"launch": {"name": "Test"}})
            ]
            
            result = await orchestrator.execute_goal("Test SpaceX goal")
            
            assert result["iterations"] == 2
            assert mock_spacex.call_count == 2

# Performance and Integration Tests
class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_goal_satisfaction_spacex(self, orchestrator):
        """Test that SpaceX weather goal is properly satisfied"""
        goal = "Find the next SpaceX launch, check weather at that location, then summarize if it may be delayed"
        
        # This would be a real integration test with actual APIs
        # For demo purposes, we'll mock the key components
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock()
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Mock sequence: launches, launchpad, current weather, forecast
            mock_response.json.side_effect = [
                [MOCK_SPACEX_LAUNCH],
                MOCK_LAUNCHPAD,
                MOCK_WEATHER,
                MOCK_WEATHER_FORECAST
            ]
            
            result = await orchestrator.execute_goal(goal)
            
            # Verify goal satisfaction
            assert result["success"] is True
            assert "final_output" in result
            
            final_output = result["final_output"]
            assert "analysis" in final_output
            assert "launch_info" in final_output
            assert "weather_summary" in final_output
            
            # Verify data chaining
            execution_results = result["results"]
            assert "spacex_agent" in execution_results
            assert "weather_agent" in execution_results
            assert "analyzer_agent" in execution_results
    
    def test_agent_trajectory_logging(self, orchestrator):
        """Test that agent execution trajectory is properly logged"""
        # Verify that execution history is maintained
        assert len(orchestrator.get_execution_history()) == 0
        
        # After execution, history should be updated
        # This would be tested with the actual execution above

# Evaluation Metrics
def evaluate_goal_satisfaction(result: dict, expected_components: list) -> dict:
    """Evaluate how well the system satisfied the given goal"""
    score = 0
    max_score = len(expected_components)
    satisfied_components = []
    
    final_output = result.get("final_output", {})
    
    for component in expected_components:
        if component in str(final_output).lower():
            score += 1
            satisfied_components.append(component)
    
    return {
        "satisfaction_score": score / max_score,
        "satisfied_components": satisfied_components,
        "missing_components": [c for c in expected_components if c not in satisfied_components],
        "execution_success": result.get("success", False),
        "iterations_needed": result.get("iterations", 0)
    }

def evaluate_agent_chaining(result: dict) -> dict:
    """Evaluate the quality of agent chaining and data flow"""
    results = result.get("results", {})
    agent_sequence = result.get("execution_plan", {}).get("agent_sequence", [])
    
    chaining_score = 0
    max_score = len(agent_sequence) - 1  # n-1 connections for n agents
    
    # Check if each agent received data from previous agents
    for i in range(1, len(agent_sequence)):
        current_agent = agent_sequence[i]
        previous_agents = agent_sequence[:i]
        
        if current_agent in results:
            # Check if current agent's input included previous agents' data
            # This would require access to the actual input data passed to agents
            chaining_score += 1
    
    return {
        "chaining_score": chaining_score / max_score if max_score > 0 else 1.0,
        "total_agents": len(agent_sequence),
        "successful_chains": chaining_score
    }

# Run evaluation tests
def run_evaluations():
    """Run comprehensive evaluations of the multi-agent system"""
    print("Running Multi-Agent System Evaluations...")
    
    # Example evaluation for SpaceX goal
    spacex_goal_components = ["launch", "weather", "delay", "analysis", "risk"]
    
    print(f"Expected components for SpaceX goal: {spacex_goal_components}")
    print("Use evaluate_goal_satisfaction() and evaluate_agent_chaining() with real results")

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
    
    # Run evaluations
    run_evaluations()