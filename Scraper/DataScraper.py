import json
import logging
import asyncio
import os
from dotenv import load_dotenv

import agentql
from playwright.async_api import async_playwright
from IPython.display import HTML
from agentql.tools.async_api import paginate

# Load environment variables from .env file
load_dotenv()
# Get the API key from the environment
AGENTQL_API_KEY = os.getenv("AGENTQL_API_KEY")
if not AGENTQL_API_KEY:
    raise Exception("AGENTQL_API_KEY not found in environment variables. Please check your .env file.")

# Configure AgentQL with your API key.
# The exact method to set the API key may vary depending on your AgentQL version.
agentql.api_key = AGENTQL_API_KEY

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context()
        page = await agentql.wrap_async(await context.new_page())
        await page.goto("https://www.shl.com/solutions/products/product-catalog/")

        # Define the query to extract post titles
        QUERY = """
        {
          Individual_Test_Solutions[]{
            name
            url
            remote_testing_support(if green circle then yes else no)
            adaptive_support(if green circle then yes else no)
            test_type
          }
        }
        """
        ##uncomment this for prepackaged_solutions
        # Query = """
        # {
        #     pre_packaged_solutions[]{
        #     name
        #     url
        #     remote_testing_support(if green circle then yes else no)
        #     adaptive_support(if green circle then yes else no)
        #     test_type
        #   }
            
            
            
        #     }""" 
       

        # Collect all data over the next 32 pages with the query defined above
        paginated_data = await paginate(page, QUERY, 32)

        # Save the aggregated data to a json file
        with open("data/shl_individual_test_solutions.json", "w") as f:
            json.dump(paginated_data, f, indent=4)
        # with open("data/shl_pre_packaged_data.json", "w") as f:
        #     json.dump(paginated_data, f, indent=4)

        log.debug("Paginated data has been saved to shl_individual_test_solutions.json")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
