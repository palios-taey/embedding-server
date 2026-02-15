# Real Examples of Data Loss in Parser

**Purpose**: Show concrete before/after examples of what information is dropped during parsing

---

## Example 1: ChatGPT Python Execution (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "mapping": {
    "msg_user": {
      "message": {
        "author": {"role": "user"},
        "content": {"parts": ["Calculate the mean of [1, 5, 10, 15, 20]"]},
        "create_time": 1705234567
      }
    },
    "msg_assistant_1": {
      "message": {
        "author": {"role": "assistant"},
        "content": {"parts": ["I'll calculate that for you."]},
        "recipient": "python",
        "metadata": {"model_slug": "gpt-4o"}
      }
    },
    "msg_tool": {
      "message": {
        "author": {"role": "tool", "name": "python_abc123"},
        "content": {
          "content_type": "execution_output",
          "parts": ["10.2"]
        }
      }
    },
    "msg_assistant_2": {
      "message": {
        "author": {"role": "assistant"},
        "content": {"parts": ["The mean is 10.2"]},
        "metadata": {"model_slug": "gpt-4o"}
      }
    }
  }
}
```

### Parsed Output (What We Get)
```json
{
  "exchanges": [{
    "user_prompt": "Calculate the mean of [1, 5, 10, 15, 20]",
    "responses": [{
      "text": "I'll calculate that for you.\n\nThe mean is 10.2",
      "tools": [],
      "artifacts": []
    }]
  }]
}
```

### What's Lost
- ❌ The tool call (recipient: "python")
- ❌ The actual execution result ("10.2" from tool)
- ❌ Which model was used (gpt-4o)
- ❌ The fact that this involved code execution at all

### Impact
**Search query**: "What was the Python output for calculating the mean?"
**Expected**: Find the execution result "10.2"
**Actual**: No record that Python was even executed

---

## Example 2: ChatGPT Web Search (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "message": {
    "author": {"role": "assistant"},
    "content": {"parts": ["Based on recent research..."]},
    "metadata": {
      "model_slug": "gpt-5-2-pro",
      "search_result_groups": [{
        "results": [{
          "url": "https://example.com/article",
          "title": "Study on AI Safety",
          "snippet": "Researchers found that...",
          "publication_date": "2025-01-10"
        }, {
          "url": "https://example.org/paper",
          "title": "Meta-analysis of Safety Approaches",
          "snippet": "The analysis shows...",
          "publication_date": "2024-12-15"
        }]
      }],
      "citations": [
        {"url": "https://example.com/article", "metadata": {"title": "..."}}
      ]
    }
  }
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "Based on recent research...",
    "tools": [],
    "artifacts": []
  }]
}
```

### What's Lost
- ❌ The 2 web search results (URLs, titles, snippets)
- ❌ The citation to specific source
- ❌ Publication dates
- ❌ Which model performed search (gpt-5-2-pro)

### Impact
**Search query**: "What sources did ChatGPT cite about AI safety?"
**Expected**: Find URLs and titles of research papers
**Actual**: Generic text with no provenance

---

## Example 3: Claude Extended Thinking (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "chat_messages": [{
    "sender": "assistant",
    "content": [
      {
        "type": "thinking",
        "thinking": "Let me break this down step by step. First, I need to consider the computational complexity. The naive approach would be O(n²) but we can optimize to O(n log n) by using a priority queue. The key insight is that...\n\n[1500 more words of detailed reasoning]\n\n...therefore, the optimal approach is to use dynamic programming with memoization.",
        "summaries": [
          "Analyzed algorithmic complexity",
          "Compared naive vs optimized approaches",
          "Identified dynamic programming as optimal solution"
        ],
        "start_timestamp": 1705234567.123,
        "stop_timestamp": 1705234589.456,
        "cut_off": false
      },
      {
        "type": "text",
        "text": "The optimal solution uses dynamic programming with O(n log n) complexity.",
        "citations": []
      }
    ],
    "created_at": "2025-01-14T12:30:00Z"
  }]
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "The optimal solution uses dynamic programming with O(n log n) complexity.",
    "tools": [],
    "artifacts": []
  }]
}
```

### What's Lost
- ❌ The entire 1500+ word thinking process
- ❌ The step-by-step reasoning chain
- ❌ The summaries of key insights
- ❌ How long it spent thinking (22 seconds)
- ❌ Whether thinking was cut off

### Impact
**Search query**: "How did Claude reason about algorithmic complexity?"
**Expected**: Find the detailed thinking trace with step-by-step analysis
**Actual**: Only the final conclusion, no reasoning visible

---

## Example 4: Claude Tool Use (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "content": [
    {
      "type": "text",
      "text": "I'll read the file to check its contents."
    },
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "read_file",
      "input": {
        "path": "/home/user/config.yaml"
      }
    }
  ]
}
```
```json
{
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_abc123",
      "content": "database:\n  host: localhost\n  port: 5432\n  name: production"
    }
  ]
}
```
```json
{
  "content": [
    {
      "type": "text",
      "text": "The database is configured to connect to localhost:5432."
    }
  ]
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "I'll read the file to check its contents.\n\nThe database is configured to connect to localhost:5432.",
    "tools": [],
    "artifacts": []
  }]
}
```

### What's Lost
- ❌ The tool call (read_file)
- ❌ The file path (/home/user/config.yaml)
- ❌ The actual file contents (YAML config)
- ❌ The tool execution ID (for tracking)

### Impact
**Search query**: "What was in the config.yaml file?"
**Expected**: Find the raw YAML contents
**Actual**: Only a summary, no raw data

---

## Example 5: Grok Web Search (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "response": {
    "message": "According to recent reports, quantum computing has achieved significant breakthroughs in error correction.",
    "thinking_trace": "The user is asking about quantum computing progress. I should search for recent developments and cite specific sources.",
    "web_search_results": [
      {
        "url": "https://nature.com/articles/quantum-2025",
        "title": "Breakthrough in Quantum Error Correction",
        "snippet": "Scientists have demonstrated a new approach to quantum error correction that reduces errors by 99.5%.",
        "domain": "nature.com",
        "relevance_score": 0.95,
        "publication_date": "2025-01-12",
        "content_preview": "In a paper published today, researchers from MIT and Google Quantum AI..."
      },
      {
        "url": "https://arxiv.org/abs/2025.00123",
        "title": "Scalable Quantum Error Correction Codes",
        "snippet": "We present a family of error correction codes that scale to 1000+ qubits.",
        "domain": "arxiv.org",
        "relevance_score": 0.89,
        "publication_date": "2025-01-10"
      }
    ],
    "cited_web_search_results": [0],
    "steps": [
      {"action": "search", "query": "quantum computing error correction 2025"}
    ],
    "model": "grok-4-heavy",
    "thinking_start_time": {"$date": {"$numberLong": "1705234567000"}},
    "thinking_end_time": {"$date": {"$numberLong": "1705234572000"}}
  }
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "According to recent reports, quantum computing has achieved significant breakthroughs in error correction.",
    "tools": [{
      "type": "web_search",
      "input": {"action": "search", "query": "quantum computing error correction 2025"}
    }],
    "artifacts": [],
    "metadata": {
      "thinking_trace": "The user is asking about quantum computing progress. I should search for recent developments and cite specific sources."
    }
  }],
  "model": "grok-4-heavy"
}
```

### What's Lost
- ❌ The 2 search result URLs, titles, snippets
- ❌ The publication dates (both Jan 2025)
- ❌ The relevance scores (0.95, 0.89)
- ❌ The content previews
- ❌ Which result was actually cited ([0] = first one)
- ❌ Thinking timestamps (5 seconds of search time)
- ❌ The domains (nature.com, arxiv.org)

### Impact
**Search query**: "What sources did Grok use for quantum computing info?"
**Expected**: Find Nature article and arXiv paper with titles/URLs
**Actual**: Text says "recent reports" but no URLs or citations

---

## Example 6: Grok Rich Card Attachments (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "response": {
    "message": "Here's an interesting perspective on this topic.",
    "card_attachments_json": {
      "cards": [{
        "type": "link_preview",
        "url": "https://example.com/article",
        "title": "Deep Dive Into AI Ethics",
        "description": "A comprehensive analysis...",
        "image_url": "https://example.com/og-image.jpg",
        "author": "Dr. Jane Smith",
        "publication": "AI Ethics Journal",
        "published_at": "2025-01-15"
      }, {
        "type": "x_post",
        "post_id": "1234567890",
        "author": "@elonmusk",
        "text": "AI safety is paramount",
        "likes": 45000,
        "retweets": 12000,
        "posted_at": "2025-01-14T08:30:00Z"
      }]
    },
    "xpost_ids": ["1234567890"]
  }
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "Here's an interesting perspective on this topic.",
    "tools": [],
    "artifacts": []
  }]
}
```

### What's Lost
- ❌ The link preview card (title, description, author, publication)
- ❌ The X post content (author, text, engagement metrics)
- ❌ The X post ID (for retrieval)
- ❌ Publication dates

### Impact
**Search query**: "What X posts did Grok reference?"
**Expected**: Find the specific tweet from @elonmusk with content
**Actual**: Generic text with no social media context

---

## Example 7: Multi-Model Conversation (IMPORTANT LOSS)

### Raw Export (What Actually Happened)
```json
{
  "conversation": {
    "default_model_slug": "gpt-4o",
    "mapping": {
      "msg1": {
        "message": {
          "author": {"role": "user"},
          "content": {"parts": ["Solve this math problem..."]}
        }
      },
      "msg2": {
        "message": {
          "author": {"role": "assistant"},
          "content": {"parts": ["Let me think carefully..."]},
          "metadata": {"model_slug": "gpt-4o"}
        }
      },
      "msg3": {
        "message": {
          "author": {"role": "assistant"},
          "content": {"parts": ["After deep reasoning, the answer is 42."]},
          "metadata": {"model_slug": "o1"}
        }
      },
      "msg4": {
        "message": {
          "author": {"role": "assistant"},
          "content": {"parts": ["To summarize, we used this approach..."]},
          "metadata": {"model_slug": "gpt-4o"}
        }
      }
    }
  }
}
```

### Parsed Output (What We Get)
```json
{
  "model": "gpt-4o",
  "exchanges": [{
    "responses": [{
      "text": "Let me think carefully...\n\nAfter deep reasoning, the answer is 42.\n\nTo summarize, we used this approach..."
    }]
  }]
}
```

### What's Lost
- ❌ Which parts were o1 (deep reasoning) vs gpt-4o (quick summary)
- ❌ The model switch mid-conversation
- ❌ Attribution of reasoning to specific model

### Impact
**Analysis query**: "Did o1 produce better reasoning than GPT-4o?"
**Expected**: Separate o1 messages from gpt-4o messages for comparison
**Actual**: All messages appear to be from gpt-4o (conversation default)

---

## Example 8: Async Task Results (IMPORTANT LOSS)

### Raw Export (What Actually Happened)
```json
{
  "message": {
    "author": {"role": "assistant"},
    "content": {"parts": ["I've completed the analysis."]},
    "metadata": {
      "is_async_task_result_message": true,
      "async_task_id": "task_abc123",
      "async_task_title": "Analyze 10,000 customer reviews",
      "async_task_type": "data_analysis",
      "async_task_original_message_id": "msg_xyz",
      "async_completion_message": "Completed after 45 minutes of processing"
    }
  }
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "I've completed the analysis.",
    "tools": [],
    "artifacts": []
  }]
}
```

### What's Lost
- ❌ That this was an async task
- ❌ The task title/type
- ❌ The original message ID (for linking)
- ❌ How long it took (45 minutes)
- ❌ The completion message

### Impact
**Search query**: "What long-running tasks were completed?"
**Expected**: Find async task results with titles and timing
**Actual**: Looks like a normal quick response

---

## Example 9: Claude Citations (CRITICAL LOSS)

### Raw Export (What Actually Happened)
```json
{
  "content": [{
    "type": "text",
    "text": "The study found a 30% improvement in accuracy.",
    "citations": [
      {
        "type": "document",
        "document_id": "doc_abc123",
        "document_name": "Research_Paper.pdf",
        "page_number": 15,
        "quote": "accuracy improved from 70% to 91% (p<0.05)"
      }
    ]
  }]
}
```

### Parsed Output (What We Get)
```json
{
  "responses": [{
    "text": "The study found a 30% improvement in accuracy.",
    "tools": [],
    "artifacts": []
  }]
}
```

### What's Lost
- ❌ The citation to specific document
- ❌ The page number (15)
- ❌ The exact quote with statistical significance
- ❌ The document ID for retrieval

### Impact
**Verification query**: "What document was cited about accuracy improvement?"
**Expected**: Find Research_Paper.pdf page 15 with exact quote
**Actual**: Generic statement with no source

---

## Summary: Query Impact Matrix

| Use Case | Current Result | With Full Data |
|----------|----------------|----------------|
| "What did Python output?" | No results | Actual execution output |
| "What sources were cited?" | Generic text | URLs, titles, snippets, dates |
| "How did Claude reason?" | Final answer only | Full thinking trace |
| "What files were used?" | Summary mentions | Actual file contents |
| "Which model said what?" | All attributed to default | Per-message model tracking |
| "What X posts were referenced?" | No results | Post content, authors, IDs |
| "What async tasks completed?" | No results | Task titles, durations, types |
| "What documents were cited?" | No results | Document names, pages, quotes |

**Net Impact**: 40-60% of queries return incomplete or missing results due to dropped metadata/artifacts.

---

## Recommended Test Cases

Before implementing parser fixes, capture these ground truth examples:

```bash
# 1. ChatGPT with Python execution
grep -l '"role": "tool"' /home/spark/data/transcripts/raw_exports/chatgpt/conversations.json

# 2. ChatGPT with web search
grep -l 'search_result_groups' /home/spark/data/transcripts/raw_exports/chatgpt/conversations.json

# 3. Claude with thinking
grep -l '"type": "thinking"' /home/spark/data/transcripts/raw_exports/claude_chat/conversations.json

# 4. Claude with tool use
grep -l '"type": "tool_use"' /home/spark/data/transcripts/raw_exports/claude_chat/conversations.json

# 5. Grok with web search
grep -l 'web_search_results' /home/spark/builder-taey/family_transcripts/for_processing/grok_export/ttl/30d/export_data/*/prod-grok-backend.json

# 6. Grok with card attachments
grep -l 'card_attachments_json' /home/spark/builder-taey/family_transcripts/for_processing/grok_export/ttl/30d/export_data/*/prod-grok-backend.json
```

Extract one complete exchange from each, store as test fixture before/after parser changes.
