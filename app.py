import streamlit as st
import anthropic
import os

class SystemPromptInfluenceAnalyzer:
    def __init__(self, api_key=None):
        """
        Initialize the analyzer with Anthropic API

        Args:
            api_key (str, optional): Anthropic API key
        """
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )

    def analyze_system_prompt_influence(
        self,
        system_prompt,
        conversation_log,
        verbose=True
    ):
        """
        Analyze the influence of system prompt segments on conversation

        Args:
            system_prompt (str): Full system prompt
            conversation_log (str): Conversation log text
            verbose (bool): Print detailed analysis

        Returns:
            dict: Analysis of system prompt segment influences
        """
        # Construct a detailed analysis prompt
        analysis_prompt = f"""
        You are an expert in system prompt interpretability and discourse analysis.

        Task: Carefully analyze the following system prompt and conversation log.
        Identify which specific segments of the system prompt directly influenced
        the agent's responses.

        System Prompt:
        {system_prompt}

        Conversation Log:
        {conversation_log}

        For EACH agent response, provide:
        1. Relevant System Prompt Segments (quote exact text)
        2. Influence Score (0-1.0)
        3. Specific Evidence of Influence
        4. Explanation of Semantic Connection

        Response Format:
        ```
        Response 1:
        - Relevant Segments: [list of segments]
        - Influence Score: X.XX
        - Evidence: [direct quote mapping]
        - Explanation: [semantic connection details]
        ```

        Provide a comprehensive, analytical breakdown that shows
        how the system prompt guides the agent's communication strategy.
        """

        try:
            # Make API call to Claude
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ]
            )

            # Extract and process the analysis
            full_analysis = response.content[0].text

            return {
                'raw_analysis': full_analysis,
                'model_used': response.model
            }

        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None

def main():
    st.title("System Prompt Influence Analyzer")

    # Sidebar for API Key input
    st.sidebar.header("Anthropic API Configuration")
    api_key = st.sidebar.text_input("Enter Anthropic API Key", type="password")

    # File uploaders
    st.header("Upload Files")
    system_prompt_file = st.file_uploader("Upload System Prompt Text File", type=['txt'])
    conversation_log_file = st.file_uploader("Upload Conversation Log Text File", type=['txt'])

    # Analysis button
    if st.button("Analyze System Prompt Influence"):
        # Validate file uploads
        if not system_prompt_file or not conversation_log_file:
            st.warning("Please upload both system prompt and conversation log files.")
            return

        # Read file contents
        try:
            system_prompt = system_prompt_file.getvalue().decode('utf-8')
            conversation_log = conversation_log_file.getvalue().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading files: {e}")
            return

        # Validate API key
        if not api_key:
            st.warning("Please enter your Anthropic API Key in the sidebar.")
            return

        # Perform analysis
        try:
            # Initialize Analyzer with provided API key
            analyzer = SystemPromptInfluenceAnalyzer(api_key)

            # Perform analysis
            st.info("Analyzing system prompt influence... This may take a few moments.")
            influence_analysis = analyzer.analyze_system_prompt_influence(
                system_prompt,
                conversation_log
            )

            # Display results
            if influence_analysis:
                st.header("Analysis Results")
                st.subheader("Raw Analysis")
                st.text_area("System Prompt Influence Analysis", 
                             value=influence_analysis['raw_analysis'], 
                             height=400)
                
                st.subheader("Model Information")
                st.write(f"Model Used: {influence_analysis['model_used']}")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    # Additional guidance
    st.sidebar.markdown("""
    ### Instructions
    1. Enter your Anthropic API Key
    2. Upload system prompt text file
    3. Upload conversation log text file
    4. Click "Analyze System Prompt Influence"

    #### File Format Requirements
    - Text files (.txt)
    - System prompt should describe the AI's behavior
    - Conversation log should include full interaction
    """)

if __name__ == "__main__":
    main()
