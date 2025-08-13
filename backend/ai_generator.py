import google.generativeai as genai
from typing import List, Optional, Dict, Any
import json

class AIGenerator:
    """Handles interactions with Google Gemini API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=800,
            ),
            system_instruction=self.SYSTEM_PROMPT
        )
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (converted to Gemini format)
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build context with conversation history
        context = ""
        if conversation_history:
            context = f"Previous conversation:\n{conversation_history}\n\n"
        
        # Prepare the full prompt
        full_prompt = f"{context}User question: {query}"
        
        # Convert tools to Gemini format if provided
        gemini_tools = None
        if tools:
            gemini_tools = self._convert_tools_to_gemini_format(tools)
        
        try:
            # Generate response with tools if available
            if gemini_tools and tool_manager:
                return self._generate_with_tools(full_prompt, gemini_tools, tool_manager)
            else:
                # Generate without tools
                response = self.model.generate_content(full_prompt)
                return response.text
                
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _convert_tools_to_gemini_format(self, anthropic_tools: List[Dict]) -> List:
        """Convert Anthropic tool format to Gemini function calling format"""
        gemini_tools = []
        
        for tool in anthropic_tools:
            if tool.get("type") == "function":
                function_def = tool.get("function", {})
                
                # Convert to Gemini format
                gemini_function = genai.protos.FunctionDeclaration(
                    name=function_def.get("name", ""),
                    description=function_def.get("description", ""),
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            prop_name: genai.protos.Schema(
                                type=self._convert_type(prop_info.get("type", "string")),
                                description=prop_info.get("description", "")
                            )
                            for prop_name, prop_info in function_def.get("parameters", {}).get("properties", {}).items()
                        },
                        required=function_def.get("parameters", {}).get("required", [])
                    )
                )
                gemini_tools.append(gemini_function)
        
        return [genai.protos.Tool(function_declarations=gemini_tools)] if gemini_tools else []
    
    def _convert_type(self, anthropic_type: str) -> genai.protos.Type:
        """Convert Anthropic parameter types to Gemini types"""
        type_mapping = {
            "string": genai.protos.Type.STRING,
            "integer": genai.protos.Type.INTEGER,
            "number": genai.protos.Type.NUMBER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT
        }
        return type_mapping.get(anthropic_type.lower(), genai.protos.Type.STRING)
    
    def _generate_with_tools(self, prompt: str, tools: List, tool_manager) -> str:
        """Generate response with function calling capability"""
        try:
            # Start chat with tools
            chat = self.model.start_chat(tools=tools)
            response = chat.send_message(prompt)
            
            # Check if the model wants to use a function
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Execute the function call
                        function_name = part.function_call.name
                        function_args = dict(part.function_call.args)
                        
                        # Execute tool through tool manager
                        tool_result = tool_manager.execute_tool(function_name, **function_args)
                        
                        # Send the function result back to the model
                        function_response = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_name,
                                response={"result": tool_result}
                            )
                        )
                        
                        # Get final response
                        final_response = chat.send_message(function_response)
                        return final_response.text
            
            # If no function call, return the direct response
            return response.text
            
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"