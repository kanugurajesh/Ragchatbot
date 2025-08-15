import google.generativeai as genai
from typing import List, Optional, Dict, Any
import json

class AIGenerator:
    """Handles interactions with Google Gemini API for generating responses"""
    
    # Simplified system prompt to avoid safety issues
    SYSTEM_PROMPT = """You are a helpful educational assistant that helps with course materials and content questions. 

You can search course content and provide course outlines when asked. Always be helpful, accurate, and concise in your responses."""
    
    def __init__(self, api_key: str, model: str):
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Configure safety settings to be minimal for educational content
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=800,
            ),
            safety_settings=safety_settings,
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
                # Configure safety settings for non-tool responses
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                
                # Generate without tools
                response = self.model.generate_content(full_prompt, safety_settings=safety_settings)
                return self._extract_response_text(response)
                
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
            # Configure safety settings for chat as well
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Start chat with tools
            chat = self.model.start_chat(tools=tools)
            response = chat.send_message(prompt, safety_settings=safety_settings)
            
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
                        final_response = chat.send_message(function_response, safety_settings=safety_settings)
                        return self._extract_response_text(final_response)
            
            # If no function call, return the direct response
            return self._extract_response_text(response)
            
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _extract_response_text(self, response) -> str:
        """
        Safely extract text from Gemini response, handling cases where response is blocked.
        
        Args:
            response: Gemini response object
            
        Returns:
            Response text or appropriate error message
        """
        try:
            # Check if response has candidates
            if not response.candidates:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            candidate = response.candidates[0]
            
            # Check finish reason for safety blocks
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason == 12:  # SAFETY
                    # Get more details about the safety ratings
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    blocked_categories = []
                    for rating in safety_ratings:
                        if hasattr(rating, 'category') and hasattr(rating, 'probability'):
                            if rating.probability.name in ['HIGH', 'MEDIUM']:
                                blocked_categories.append(rating.category.name)
                    
                    if blocked_categories:
                        return f"Response blocked due to safety filters. Categories: {', '.join(blocked_categories)}. This appears to be a false positive for educational content."
                    else:
                        return "Response blocked by safety filters. This appears to be a false positive for educational content."
                elif candidate.finish_reason == 3:  # RECITATION
                    return "Response blocked due to potential recitation. Please try rephrasing your question."
                elif candidate.finish_reason not in [1, None]:  # 1 = STOP (normal completion)
                    return f"Response incomplete (finish_reason: {candidate.finish_reason}). Please try again."
            
            # Check if content exists and has parts
            if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                # Extract text from parts
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                
                if text_parts:
                    return ''.join(text_parts)
            
            # Fallback: try direct text access
            if hasattr(response, 'text') and response.text:
                return response.text
            
            return "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing the response: {str(e)}"