# npm package

The [mindstack npm package](https://www.npmjs.com/package/gpt-researcher) is a WebSocket client for interacting with MindStack.

## Installation

```bash
npm install mindstack
```

## Usage

```javascript
const GPTResearcher = require('mindstack');

const researcher = new GPTResearcher({
  host: 'localhost:8000',
  logListener: (data) => console.log('logListener logging data: ',data)
});

researcher.sendMessage({
  query: 'Does providing better context reduce LLM hallucinations?',
  moreContext: 'Provide a detailed answer'
});
```
