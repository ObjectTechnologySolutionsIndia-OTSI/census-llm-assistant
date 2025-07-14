import asyncio

# from utils.vector_indexing.indexing_process import main
from utils.vector_indexing.indexing_ingestion_batch_async import AsyncEmbeddingProcessor, logger



async def main():
    """Example usage of the AsyncEmbeddingProcessor."""
    
    # Create sample data (300 records to demonstrate batching)
    # input_data = []
    # base_texts = [
    #     "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    #     "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    #     "Natural language processing enables computers to understand, interpret, and generate human language.",
    #     "Computer vision allows machines to interpret and understand visual information from the world.",
    #     "Reinforcement learning is a type of machine learning where agents learn to make decisions through trial and error.",
    #     "Data science combines statistical analysis, machine learning, and domain expertise to extract insights from data.",
    #     "Artificial neural networks are computing systems inspired by biological neural networks.",
    #     "Big data refers to large, complex datasets that require special tools and techniques to process.",
    #     "Cloud computing provides on-demand access to computing resources over the internet.",
    #     "Blockchain is a distributed ledger technology that maintains a continuously growing list of records."
    # ]
    
    # # Generate 250 records for demonstration
    # for i in range(250):
    #     text_index = i % len(base_texts)
    #     input_data.append({
    #         "chunk": f"{base_texts[text_index]} (Record {i+1})",
    #         "metadata_text": f"Sample metadata for record {i+1}",
    #         "source_file": f"document_{(i//10)+1}.pdf"
    #     })
    
    input_data =  [
    {
        "chunk": "In 'INDIA' ('INDIA @&'), the 'Total' area has 597608.0 inhabited villages, 43324.0 uninhabited villages, 7933.0 towns, 249501663.0 households, a total population of 1210854977.0 (males: 623270258.0, females: 587584719.0), covering an area of 3287469.0 sq. km with a population density of 382.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'INDIA' ('INDIA $'), the 'Rural' area has 597608.0 inhabited villages, 43324.0 uninhabited villages, 0.0 towns, 168612897.0 households, a total population of 833748852.0 (males: 427781058.0, females: 405967794.0), covering an area of 3101473.97 sq. km with a population density of 279.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'INDIA' ('INDIA $'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 7933.0 towns, 80888766.0 households, a total population of 377106125.0 (males: 195489200.0, females: 181616925.0), covering an area of 102252.03 sq. km with a population density of 3685.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'STATE' ('JAMMU & KASHMIR @&'), the 'Total' area has 6337.0 inhabited villages, 216.0 uninhabited villages, 122.0 towns, 2119718.0 households, a total population of 12541302.0 (males: 6640662.0, females: 5900640.0), covering an area of 222236.0 sq. km with a population density of 124.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'STATE' ('JAMMU & KASHMIR'), the 'Rural' area has 6337.0 inhabited villages, 216.0 uninhabited villages, 0.0 towns, 1553433.0 households, a total population of 9108060.0 (males: 4774477.0, females: 4333583.0), covering an area of 220990.1 sq. km with a population density of 91.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'STATE' ('JAMMU & KASHMIR'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 122.0 towns, 566285.0 households, a total population of 3433242.0 (males: 1866185.0, females: 1567057.0), covering an area of 1245.9 sq. km with a population density of 2755.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Kupwara'), the 'Total' area has 353.0 inhabited villages, 9.0 uninhabited villages, 10.0 towns, 113929.0 households, a total population of 870354.0 (males: 474190.0, females: 396164.0), covering an area of 2379.0 sq. km with a population density of 366.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Kupwara'), the 'Rural' area has 353.0 inhabited villages, 9.0 uninhabited villages, 0.0 towns, 101930.0 households, a total population of 765625.0 (males: 412038.0, females: 353587.0), covering an area of 2331.66 sq. km with a population density of 328.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Kupwara'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 10.0 towns, 11999.0 households, a total population of 104729.0 (males: 62152.0, females: 42577.0), covering an area of 47.34 sq. km with a population density of 2212.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Kupwara'), the 'Total' area has 118.0 inhabited villages, 4.0 uninhabited villages, 7.0 towns, 63022.0 households, a total population of 540914.0 (males: 297837.0, females: 243077.0), covering an area of 301.94 sq. km with a population density of 1791.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Kupwara'), the 'Rural' area has 118.0 inhabited villages, 4.0 uninhabited villages, 0.0 towns, 56014.0 households, a total population of 465323.0 (males: 252856.0, females: 212467.0), covering an area of 275.03 sq. km with a population density of 1692.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Kupwara'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 7.0 towns, 7008.0 households, a total population of 75591.0 (males: 44981.0, females: 30610.0), covering an area of 26.91 sq. km with a population density of 2809.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Handwara'), the 'Total' area has 196.0 inhabited villages, 3.0 uninhabited villages, 1.0 towns, 39485.0 households, a total population of 269311.0 (males: 141882.0, females: 127429.0), covering an area of 291.47 sq. km with a population density of 924.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Handwara'), the 'Rural' area has 196.0 inhabited villages, 3.0 uninhabited villages, 0.0 towns, 37474.0 households, a total population of 255711.0 (males: 134503.0, females: 121208.0), covering an area of 282.97 sq. km with a population density of 904.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Handwara'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 1.0 towns, 2011.0 households, a total population of 13600.0 (males: 7379.0, females: 6221.0), covering an area of 8.5 sq. km with a population density of 1600.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Karnah'), the 'Total' area has 39.0 inhabited villages, 2.0 uninhabited villages, 2.0 towns, 11422.0 households, a total population of 60129.0 (males: 34471.0, females: 25658.0), covering an area of 69.89 sq. km with a population density of 860.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Karnah'), the 'Rural' area has 39.0 inhabited villages, 2.0 uninhabited villages, 0.0 towns, 8442.0 households, a total population of 44591.0 (males: 24679.0, females: 19912.0), covering an area of 57.96 sq. km with a population density of 769.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Karnah'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 2.0 towns, 2980.0 households, a total population of 15538.0 (males: 9792.0, females: 5746.0), covering an area of 11.93 sq. km with a population density of 1302.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Badgam'), the 'Total' area has 462.0 inhabited villages, 12.0 uninhabited villages, 9.0 towns, 103363.0 households, a total population of 753745.0 (males: 398041.0, females: 355704.0), covering an area of 1361.0 sq. km with a population density of 554.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Badgam'), the 'Rural' area has 462.0 inhabited villages, 12.0 uninhabited villages, 0.0 towns, 89417.0 households, a total population of 655833.0 (males: 343385.0, females: 312448.0), covering an area of 1311.95 sq. km with a population density of 500.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Badgam'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 9.0 towns, 13946.0 households, a total population of 97912.0 (males: 54656.0, females: 43256.0), covering an area of 49.05 sq. km with a population density of 1996.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khag'), the 'Total' area has 49.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 8799.0 households, a total population of 67596.0 (males: 34457.0, females: 33139.0), covering an area of 61.12 sq. km with a population density of 1106.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khag'), the 'Rural' area has 49.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 8799.0 households, a total population of 67596.0 (males: 34457.0, females: 33139.0), covering an area of 61.12 sq. km with a population density of 1106.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khag'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 0.0 households, a total population of 0.0 (males: 0.0, females: 0.0), covering an area of 0.0 sq. km with a population density of 0.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Beerwah'), the 'Total' area has 102.0 inhabited villages, 2.0 uninhabited villages, 2.0 towns, 20914.0 households, a total population of 163333.0 (males: 87277.0, females: 76056.0), covering an area of 134.07 sq. km with a population density of 1218.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Beerwah'), the 'Rural' area has 102.0 inhabited villages, 2.0 uninhabited villages, 0.0 towns, 19161.0 households, a total population of 149671.0 (males: 80103.0, females: 69568.0), covering an area of 126.77 sq. km with a population density of 1181.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Beerwah'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 2.0 towns, 1753.0 households, a total population of 13662.0 (males: 7174.0, females: 6488.0), covering an area of 7.3 sq. km with a population density of 1872.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khansahib'), the 'Total' area has 96.0 inhabited villages, 3.0 uninhabited villages, 1.0 towns, 16432.0 households, a total population of 123312.0 (males: 64227.0, females: 59085.0), covering an area of 125.91 sq. km with a population density of 979.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khansahib'), the 'Rural' area has 96.0 inhabited villages, 3.0 uninhabited villages, 0.0 towns, 16080.0 households, a total population of 120682.0 (males: 62783.0, females: 57899.0), covering an area of 121.61 sq. km with a population density of 992.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khansahib'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 1.0 towns, 352.0 households, a total population of 2630.0 (males: 1444.0, females: 1186.0), covering an area of 4.3 sq. km with a population density of 612.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Budgam'), the 'Total' area has 95.0 inhabited villages, 2.0 uninhabited villages, 2.0 towns, 19774.0 households, a total population of 133855.0 (males: 70182.0, females: 63673.0), covering an area of 146.16 sq. km with a population density of 916.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Budgam'), the 'Rural' area has 95.0 inhabited villages, 2.0 uninhabited villages, 0.0 towns, 16517.0 households, a total population of 111056.0 (males: 57359.0, females: 53697.0), covering an area of 131.44 sq. km with a population density of 845.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Budgam'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 2.0 towns, 3257.0 households, a total population of 22799.0 (males: 12823.0, females: 9976.0), covering an area of 14.72 sq. km with a population density of 1549.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Chadoora'), the 'Total' area has 104.0 inhabited villages, 4.0 uninhabited villages, 3.0 towns, 30390.0 households, a total population of 212233.0 (males: 113529.0, females: 98704.0), covering an area of 194.67 sq. km with a population density of 1090.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Chadoora'), the 'Rural' area has 104.0 inhabited villages, 4.0 uninhabited villages, 0.0 towns, 23904.0 households, a total population of 164945.0 (males: 86218.0, females: 78727.0), covering an area of 172.74 sq. km with a population density of 955.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Chadoora'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 3.0 towns, 6486.0 households, a total population of 47288.0 (males: 27311.0, females: 19977.0), covering an area of 21.93 sq. km with a population density of 2156.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Charar- E- Shrief'), the 'Total' area has 16.0 inhabited villages, 1.0 uninhabited villages, 1.0 towns, 7054.0 households, a total population of 53416.0 (males: 28369.0, females: 25047.0), covering an area of 58.85 sq. km with a population density of 908.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Charar- E- Shrief'), the 'Rural' area has 16.0 inhabited villages, 1.0 uninhabited villages, 0.0 towns, 4956.0 households, a total population of 41883.0 (males: 22465.0, females: 19418.0), covering an area of 58.05 sq. km with a population density of 721.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Charar- E- Shrief'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 1.0 towns, 2098.0 households, a total population of 11533.0 (males: 5904.0, females: 5629.0), covering an area of 0.8 sq. km with a population density of 14416.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Leh(Ladakh)'), the 'Total' area has 111.0 inhabited villages, 1.0 uninhabited villages, 3.0 towns, 21909.0 households, a total population of 133487.0 (males: 78971.0, females: 54516.0), covering an area of 45110.0 sq. km with a population density of 3.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Leh(Ladakh)'), the 'Rural' area has 111.0 inhabited villages, 1.0 uninhabited villages, 0.0 towns, 14905.0 households, a total population of 87816.0 (males: 48411.0, females: 39405.0), covering an area of 45085.99 sq. km with a population density of 2.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'DISTRICT' ('Leh(Ladakh)'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 3.0 towns, 7004.0 households, a total population of 45671.0 (males: 30560.0, females: 15111.0), covering an area of 24.01 sq. km with a population density of 1902.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Leh'), the 'Total' area has 60.0 inhabited villages, 1.0 uninhabited villages, 3.0 towns, 15497.0 households, a total population of 93961.0 (males: 56597.0, females: 37364.0), covering an area of 171.06 sq. km with a population density of 549.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Leh'), the 'Rural' area has 60.0 inhabited villages, 1.0 uninhabited villages, 0.0 towns, 8493.0 households, a total population of 48290.0 (males: 26037.0, females: 22253.0), covering an area of 147.05 sq. km with a population density of 328.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Leh'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 3.0 towns, 7004.0 households, a total population of 45671.0 (males: 30560.0, females: 15111.0), covering an area of 24.01 sq. km with a population density of 1902.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Nubra'), the 'Total' area has 28.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 3606.0 households, a total population of 22433.0 (males: 13740.0, females: 8693.0), covering an area of 166.82 sq. km with a population density of 134.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Nubra'), the 'Rural' area has 28.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 3606.0 households, a total population of 22433.0 (males: 13740.0, females: 8693.0), covering an area of 166.82 sq. km with a population density of 134.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Nubra'), the 'Urban' area has 0.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 0.0 households, a total population of 0.0 (males: 0.0, females: 0.0), covering an area of 0.0 sq. km with a population density of 0.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khalsi'), the 'Total' area has 23.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 2806.0 households, a total population of 17093.0 (males: 8634.0, females: 8459.0), covering an area of 56.01 sq. km with a population density of 305.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    },
    {
        "chunk": "In 'SUB-DISTRICT' ('Khalsi'), the 'Rural' area has 23.0 inhabited villages, 0.0 uninhabited villages, 0.0 towns, 2806.0 households, a total population of 17093.0 (males: 8634.0, females: 8459.0), covering an area of 56.01 sq. km with a population density of 305.0 per sq. km.",
        "metadata_text": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
        "source_file": "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    }
]
       
    # Initialize the processor with batch size of 100
    processor = AsyncEmbeddingProcessor(
        voyage_model="voyage-3-large",
        voyage_input_type="document",
        batch_size=100,
        max_concurrent_operations=3
    )
    
    try:
        # Initialize components
        await processor.initialize()
        
        print(f"📊 Processing {len(input_data)} records in batches of {processor.batch_size}")
        
        # Process documents with async pipeline
        result = await processor.process_documents_async_pipeline(input_data)
        
        if result["success"]:
            print("✅ Successfully processed all documents!")
            print(f"📈 Results:")
            print(f"   - Total records: {result['total_records']}")
            print(f"   - Total batches: {result['total_batches']}")
            print(f"   - Successful batches: {result['processed_batches']}")
            print(f"   - Failed batches: {result['failed_batches']}")
            print(f"   - Successful records: {result['successful_records']}")
            
            # Get database statistics
            if processor.db_manager:
                stats = await processor.db_manager.get_table_stats()
                print(f"📊 Database statistics: {stats}")
        else:
            print("❌ Failed to process documents")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        
    finally:
        # Clean up resources
        await processor.cleanup()




if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())