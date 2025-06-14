# Multi-Agent AI System

A sophisticated multi-agent system that takes user goals, creates execution plans, and chains specialized agents to achieve complex objectives. Each agent enriches the output of the previous one, creating a powerful data pipeline that can handle diverse tasks from SpaceX launch analysis to cryptocurrency sentiment tracking.

## 🎯 Overview

This system demonstrates advanced AI orchestration by:
- **Intelligent Planning**: Automatically determines the best sequence of agents for any given goal
- **Agent Chaining**: Each agent builds upon previous results, creating rich, contextual outputs
- **Iterative Refinement**: Automatically retries failed executions up to 3 times
- **Modular Architecture**: Easy to extend with new agents and capabilities

## 🏗️ Architecture

```
User Goal → Planner Agent → Specialized Agents → Final Result
              ↓
         Execution Plan
              ↓
    [Agent 1] → [Agent 2] → [Agent 3] → Output
         ↓           ↓          ↓
    Data Flow   Enrichment  Analysis
```

### Core Components

1. **MultiAgentOrchestrator**: Main controller that manages execution flow
2. **PlannerAgent**: Creates intelligent execution plans based on user goals
3. **Specialized Agents**: Domain-specific agents that perform targeted tasks
4. **AgentResult**: Standardized data format for inter-agent communication

## 🤖 Available Agents

### Planning Agent
- **PlannerAgent**: Analyzes user goals and creates optimal execution plans

### SpaceX Pipeline Agents
- **SpaceXAgent**: Fetches upcoming launch data from SpaceX API
- **WeatherAgent**: Gets weather conditions for launch locations
- **AnalyzerAgent**: Calculates launch delay probability based on weather

### Cryptocurrency Pipeline Agents
- **CryptoAgent**: Retrieves cryptocurrency prices and market data
- **NewsAgent**: Fetches relevant news articles
- **SentimentAgent**: Analyzes news sentiment and market impact

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-ai-system

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```env
OPENWEATHER_API_KEY=your_openweather_api_key
NEWS_API_KEY=your_news_api_key
```

### Basic Usage

```python
import asyncio
from multi_agent_system import MultiAgentOrchestrator

async def main():
    # Initialize with API keys
    api_keys = {
        "OPENWEATHER_API_KEY": "your_key_here",
        "NEWS_API_KEY": "your_key_here"
    }
    
    orchestrator = MultiAgentOrchestrator(api_keys)
    
    # Execute a goal
    result = await orchestrator.execute_goal(
        "Find the next SpaceX launch and check if weather may cause delays"
    )
    
    print(f"Success: {result['success']}")
    print(f"Final Output: {result['final_output']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📋 Requirements

### Python Dependencies
```
aiohttp>=3.8.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
python-dotenv>=1.0.0
```

### API Keys Required
- **OpenWeatherMap API**: For weather data (free tier available)
- **NewsAPI**: For news articles (free tier available)
- **SpaceX API**: Public API, no key required
- **CoinGecko API**: Public API, no key required

## 🎯 Example Goals

### SpaceX Launch Analysis
```python
goal = "Find the next SpaceX launch, check weather at that location, then summarize if it may be delayed"
```
**Expected Output**: Launch details, weather conditions, delay probability analysis

### Cryptocurrency Sentiment Analysis
```python
goal = "Get Bitcoin price and recent news, then analyze market sentiment"
```
**Expected Output**: Current prices, news articles, sentiment analysis, market impact

### Custom Goals
The system intelligently routes any goal to appropriate agents:
```python
goal = "Analyze Ethereum market conditions with news sentiment"
goal = "Check weather conditions for upcoming rocket launches"
```

## 🔧 Configuration

### Adding New Agents

1. **Create Agent Class**:
```python
class MyCustomAgent(BaseAgent):
    def __init__(self, api_keys: Dict[str, str]):
        super().__init__("my_agent", api_keys)
    
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        # Implementation here
        return self._create_result({"data": "result"})
```

2. **Register in Orchestrator**:
```python
self.agents["my_agent"] = MyCustomAgent(api_keys)
```

3. **Update Planner Templates**:
```python
self.goal_templates["my_template"] = {
    "agents": ["my_agent", "other_agents"],
    "description": "Template description"
}
```

### Extending Goal Classification

Modify the `PlannerAgent.execute()` method to add new goal patterns:

```python
if "my_keyword" in goal:
    template = self.goal_templates["my_template"]
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Test individual agents
pytest tests/test_agents.py -v

# Test orchestration
pytest tests/test_orchestrator.py -v

# Test integration
pytest tests/test_integration.py -v
```

### Evaluation Metrics

The system includes comprehensive evaluation tools:

```python
from tests.evaluations import evaluate_goal_satisfaction, evaluate_agent_chaining

# Evaluate goal satisfaction
satisfaction = evaluate_goal_satisfaction(result, expected_components)
print(f"Goal satisfaction: {satisfaction['satisfaction_score']:.2%}")

# Evaluate agent chaining quality
chaining = evaluate_agent_chaining(result)
print(f"Agent chaining score: {chaining['chaining_score']:.2%}")
```

## 📊 System Flow

### 1. Goal Analysis
- User provides natural language goal
- PlannerAgent analyzes and classifies the goal
- Creates execution plan with agent sequence

### 2. Agent Execution
- Agents execute in planned sequence
- Each agent receives enriched data from previous agents
- Results are accumulated in standardized format

### 3. Error Handling & Retry
- Failed agents trigger retry mechanism
- Up to 3 iterations to achieve goal
- Graceful degradation with mock data when APIs unavailable

### 4. Result Compilation
- Final output extracted from last agent
- Complete execution history maintained
- Success/failure status and metrics tracked

## 🔍 Agent Chaining Details

### Data Flow Example (SpaceX Pipeline)
```
SpaceXAgent → WeatherAgent → AnalyzerAgent
     ↓             ↓              ↓
Launch Data → Weather Data → Risk Analysis
```

1. **SpaceXAgent** fetches launch and location data
2. **WeatherAgent** uses location to get weather conditions
3. **AnalyzerAgent** combines launch + weather data for delay analysis

### Input Data Structure
Each agent receives:
```python
{
    "goal": "original user goal",
    "previous_agent_id": AgentResult,
    "another_agent_id": AgentResult,
    # ... all previous results
}
```

## 🎛️ Advanced Features

### Intelligent Routing
- Automatic agent selection based on goal keywords
- Dependency resolution and execution ordering
- Fallback to generic pipeline for unknown goals

### Robust Error Handling
- Individual agent failure recovery
- API fallback to mock data
- Comprehensive logging and debugging

### Extensible Design
- Plugin architecture for new agents
- Standardized result formats
- Configurable retry mechanisms

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in environment
   - Check API key validity and rate limits

2. **Agent Failures**
   - Review logs for specific error messages
   - Verify internet connectivity for external APIs
   - Check if APIs are returning expected data formats

3. **Goal Not Recognized**
   - Add new goal patterns to PlannerAgent
   - Ensure keywords are properly classified

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Submit a pull request

### Development Guidelines
- Follow existing code structure and patterns
- Add proper error handling and logging
- Include unit tests for new agents
- Update documentation for new features

## 📈 Performance Considerations

- **Async Operations**: All agents use async/await for concurrent execution
- **API Rate Limits**: Built-in respect for API limitations
- **Memory Management**: Efficient data structures and cleanup
- **Retry Logic**: Intelligent backoff and retry mechanisms

## 🔐 Security Notes

- Store API keys securely in environment variables
- Never commit API keys to version control
- Implement proper input validation
- Consider rate limiting for production use

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SpaceX API for launch data
- OpenWeatherMap for weather information
- NewsAPI for news articles
- CoinGecko for cryptocurrency data

---

**Built with ❤️ by Abish**