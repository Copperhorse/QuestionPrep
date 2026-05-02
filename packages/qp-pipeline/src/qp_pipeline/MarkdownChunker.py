import re
import textwrap
import time  # <--- Added for timing
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 1. DEPENDENCY CHECKS
# ============================================================

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / 4))


# ============================================================
# 2. ENUMS & CONFIG
# ============================================================


class ContentType(Enum):
    CODE = "code"
    MATH = "math"
    TABLE = "table"
    PROSE = "prose"


@dataclass
class ContentBlock:
    content: str
    type: ContentType


@dataclass
class ChunkConfig:
    max_chunk_tokens: int = 500
    min_chunk_tokens: int = 50
    merge_short_chunks: bool = True
    use_recursive_splitter: bool = True


# ============================================================
# 3. TEXT SPLITTERS
# ============================================================


class NativeTextSplitter:
    """Recursively splits text by separators without external libs."""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)
        current_chunk = []
        current_len = 0

        for s in splits:
            s_len = count_tokens(s)
            if current_len + s_len > self.chunk_size:
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                if s_len > self.chunk_size and new_separators:
                    final_chunks.extend(self._split(s, new_separators))
                    current_chunk = []
                    current_len = 0
                else:
                    current_chunk = [s]
                    current_len = s_len
            else:
                current_chunk.append(s)
                current_len += s_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return [c.strip() for c in final_chunks if c.strip()]


# ============================================================
# 4. ROBUST CHUNKER CLASS
# ============================================================


class MarkdownChunker:
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # Patterns for identifying special blocks
        self.block_patterns = [
            (re.compile(r"```[\s\S]*?```"), ContentType.CODE),
            (re.compile(r"\$\$[\s\S]*?\$\$"), ContentType.MATH),
            # ✅ FIX: Faster, Safer Table Regex
            # This prevents the infinite hang on large tables in survey papers
            (
                re.compile(r"(?:^\|.*?\|\s*$(?:\n^\|.*?\|\s*$)+)", re.MULTILINE),
                ContentType.TABLE,
            ),
        ]

        if LANGCHAIN_AVAILABLE and self.config.use_recursive_splitter:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_chunk_tokens * 4,
                chunk_overlap=50,
                length_function=count_tokens,
            )
        else:
            self.splitter = NativeTextSplitter(chunk_size=self.config.max_chunk_tokens)

    # -----------------------------------------------------------------
    # A. MASKING UTILITIES (Protection)
    # -----------------------------------------------------------------

    def _mask_content(
        self, text: str
    ) -> Tuple[str, Dict[str, Tuple[str, ContentType]]]:
        mask_store = {}
        masked_text = text

        for pat, c_type in self.block_patterns:

            def replacer(match):
                content = match.group(0)
                mask_id = f"__MASK_{uuid.uuid4().hex[:8]}__"
                mask_store[mask_id] = (content, c_type)
                return f"\n{mask_id}\n"

            masked_text = pat.sub(replacer, masked_text)

        return masked_text, mask_store

    # -----------------------------------------------------------------
    # B. NORMALIZATION (Strict Mode)
    # -----------------------------------------------------------------

    def _normalize_headers(self, text: str) -> str:
        lines = text.split("\n")
        new_lines = []
        md_pat = re.compile(r"^\s*(#{1,6})\s+(.*)")

        for line in lines:
            line_str = line.strip()

            if not line_str or (
                line_str.startswith("__MASK_") and line_str.endswith("__")
            ):
                new_lines.append(line)
                continue

            m_md = md_pat.match(line_str)
            if m_md:
                new_lines.append(f"{m_md.group(1)} {m_md.group(2)}")
                continue

            new_lines.append(line)

        return "\n".join(new_lines)

    # -----------------------------------------------------------------
    # C. HIERARCHY SPLITTING
    # -----------------------------------------------------------------

    def _split_hierarchical_sections(self, text: str) -> List[Dict[str, Any]]:
        lines = text.split("\n")
        sections = []
        current_content = []
        header_stack = {}  # {level: title}
        current_level = 0

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.*)", line)
            is_mask = "__MASK_" in line and line.strip().endswith("__")

            if match and not is_mask:
                if current_content:
                    header_text = header_stack.get(current_level, "Root")
                    parent_text = next(
                        (
                            header_stack[k]
                            for k in range(current_level - 1, 0, -1)
                            if k in header_stack
                        ),
                        None,
                    )
                    top_text = header_stack.get(1, header_text)

                    sections.append(
                        {
                            "header": header_text,
                            "section_level": current_level,
                            "parent": parent_text,
                            "top_header": top_text,
                            "content": "\n".join(current_content).strip(),
                        }
                    )

                hashes, title = match.groups()
                new_level = len(hashes)
                current_level = new_level
                header_stack[new_level] = title.strip()

                keys_to_wipe = [k for k in header_stack if k > new_level]
                for k in keys_to_wipe:
                    del header_stack[k]

                current_content = []
            else:
                current_content.append(line)

        if current_content:
            header_text = header_stack.get(current_level, "Root")
            parent_text = next(
                (
                    header_stack[k]
                    for k in range(current_level - 1, 0, -1)
                    if k in header_stack
                ),
                None,
            )
            top_text = header_stack.get(1, header_text)

            sections.append(
                {
                    "header": header_text,
                    "section_level": current_level,
                    "parent": parent_text,
                    "top_header": top_text,
                    "content": "\n".join(current_content).strip(),
                }
            )

        return sections

    # -----------------------------------------------------------------
    # D. BLOCK PROCESSING
    # -----------------------------------------------------------------

    def _process_section_blocks(
        self, section: Dict[str, Any], mask_store: Dict[str, Tuple[str, ContentType]]
    ) -> List[Dict[str, Any]]:
        masked_text = section["content"]
        if not masked_text:
            return []

        mask_pat = re.compile(r"(__MASK_[0-9a-f]+__)")
        parts = mask_pat.split(masked_text)
        final_chunks = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part in mask_store:
                original_content, c_type = mask_store[part]

                if (
                    c_type == ContentType.TABLE
                    and count_tokens(original_content) > self.config.max_chunk_tokens
                ):
                    sub_texts = (
                        self.splitter.split_text(original_content)
                        if hasattr(self.splitter, "split_text")
                        else [original_content]
                    )
                else:
                    sub_texts = [original_content]

                for t in sub_texts:
                    final_chunks.append(
                        {
                            "content": t,
                            "metadata": {
                                "section_header": section["header"],
                                "parent_section": section["parent"],
                                "top_header": section["top_header"],
                                "content_type": c_type.value,
                            },
                            "estimated_tokens": count_tokens(t),
                        }
                    )

            else:
                if hasattr(self.splitter, "split_text"):
                    sub_texts = self.splitter.split_text(part)
                else:
                    sub_texts = [part]

                for t in sub_texts:
                    final_chunks.append(
                        {
                            "content": t,
                            "metadata": {
                                "section_header": section["header"],
                                "parent_section": section["parent"],
                                "top_header": section["top_header"],
                                "content_type": ContentType.PROSE.value,
                            },
                            "estimated_tokens": count_tokens(t),
                        }
                    )

        return final_chunks

    # -----------------------------------------------------------------
    # E. MAIN PROCESS
    # -----------------------------------------------------------------

    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        masked_text, mask_store = self._mask_content(markdown_text)
        norm_text = self._normalize_headers(masked_text)
        sections = self._split_hierarchical_sections(norm_text)

        all_chunks = []
        for sec in sections:
            all_chunks.extend(self._process_section_blocks(sec, mask_store))

        if self.config.merge_short_chunks:
            all_chunks = self._merge_chunks(all_chunks)

        return self._add_ids(all_chunks)

    def _merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        merged = []
        buffer = None

        for chunk in chunks:
            if not buffer:
                buffer = chunk
                continue

            diff_header = (
                buffer["metadata"]["section_header"]
                != chunk["metadata"]["section_header"]
            )
            diff_type = (
                buffer["metadata"]["content_type"] != chunk["metadata"]["content_type"]
            )
            too_big = (
                buffer["estimated_tokens"] + chunk["estimated_tokens"]
            ) > self.config.max_chunk_tokens

            if diff_header or diff_type or too_big:
                merged.append(buffer)
                buffer = chunk
            else:
                buffer["content"] += "\n\n" + chunk["content"]
                buffer["estimated_tokens"] += chunk["estimated_tokens"]

        if buffer:
            merged.append(buffer)
        return merged

    def _add_ids(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, c in enumerate(chunks):
            c["chunk_id"] = str(uuid.uuid4())
            c["chunk_index"] = i
            c["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            c["next_chunk_id"] = None

        for i in range(len(chunks) - 1):
            chunks[i]["next_chunk_id"] = chunks[i + 1]["chunk_id"]
        return chunks


def chunk_markdown(text: str, **kwargs) -> List[Dict[str, Any]]:
    config = ChunkConfig(**kwargs)
    return MarkdownChunker(config).process(text)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING STRICT CHUNKING (README SAMPLE)")
    print("=" * 60)

    # Use Raw String (r"") to handle Windows paths in the text
    readme_sample = textwrap.dedent(r"""
    ## A Survey on Data Collection for Machine Learning

    A Big Data - AI Integration Perspective

    Yuji Roh, Geon Heo, Steven Euijong Whang, Senior Member, IEEE

    Abstract -Data collection is a major bottleneck in machine learning and an active research topic in multiple communities. There are largely two reasons data collection has recently become a critical issue. First, as machine learning is becoming more widely-used, we are seeing new applications that do not necessarily have enough labeled data. Second, unlike traditional machine learning, deep learning techniques automatically generate features, which saves feature engineering costs, but in return may require larger amounts of labeled data. Interestingly, recent research in data collection comes not only from the machine learning, natural language, and computer vision communities, but also from the data management community due to the importance of handling large amounts of data. In this survey, we perform a comprehensive study of data collection from a data management point of view. Data collection largely consists of data acquisition, data labeling, and improvement of existing data or models. We provide a research landscape of these operations, provide guidelines on which technique to use when, and identify interesting research challenges. The integration of machine learning and data management for data collection is part of a larger trend of Big data and Artificial Intelligence (AI) integration and opens many opportunities for new research.

    Index Terms -data collection, data acquisition, data labeling, machine learning

    F

    ## 1 INTRODUCTION

    W Eareliving in exciting times where machine learning is having a profound influence on a wide range of applications from text understanding, image and speech recognition, to health care and genomics. As a striking example, deep learning techniques are known to perform on par with ophthalmologists on identifying diabetic eye diseases in images [1]. Much of the recent success is due to better computation infrastructure and large amounts of training data.

    Among the many challenges in machine learning, data collection is becoming one of the critical bottlenecks. It is known that the majority of the time for running machine learning end-to-end is spent on preparing the data, which includes collecting, cleaning, analyzing, visualizing, and feature engineering. While all of these steps are timeconsuming, data collection has recently become a challenge due to the following reasons.

    First, as machine learning is used in new applications, it is usually the case that there is not enough training data. Traditional applications like machine translation or object detection enjoy massive amounts of training data that have been accumulated for decades. On the other hand, more recent applications have little or no training data. As an illustration, smart factories are increasingly becoming automated where product quality control is performed with machine learning. Whenever there is a new product or a new defect to detect, there is little or no training data to start with. The na¨ ıve approach of manual labeling may not be feasible because it is expensive and requires domain

    · Y. Roh, G. Heo, and S. E. Whang are with the School of Electrical Engineering, KAIST, Daejeon, Korea. E-mail: { yuji.roh, geon.heo, swhang } @kaist.ac.kr

    · Corresponding author: S. E. Whang

    expertise. This problem applies to any novel application that benefits from machine learning.

    Moreover, as deep learning [2] becomes popular, there is even more need for training data. In traditional machine learning, feature engineering is one of the most challenging steps where the user needs to understand the application and provide features used for training models. Deep learning, on the other hand, can automatically generate features, which saves us of feature engineering, which is a significant part of data preparation. However, in return, deep learning may require larger amounts of training data to perform well [3].

    As a result, there is a pressing need of accurate and scalable data collection techniques in the era of Big data, which motivates us to conduct a comprehensive survey of the data collection literature from a data management point of view. There are largely three methods for data collection. First, if the goal is to share and search new datasets, then data acquisition techniques can be used to discover, augment, or generate datasets. Second, once the datasets are available, various data labeling techniques can be used to label the individual examples. Finally, instead of labeling new datasets, it may be better to improve existing data or train on top of trained models. These three methods are not necessarily distinct and can be used together. For example, one could search and label more datasets while improving existing ones.

    An interesting observation is that the data collection techniques come not only from the machine learning community (including natural language processing and computer vision, which traditionally use machine learning heavily), but have also been studied for decades by the data management community, mainly under the names of data science and data analytics. Figure 1 shows an overview of

    Fig. 1: A high level research landscape of data collection for machine learning. The topics that are at least partially contributed by the data management community are highlighted using blue italic text. Hence, to fully understand the research landscape, one needs to look at the literature from the viewpoints of both the machine learning and data management communities.

    <!-- image -->

    the research landscape where the topics that have contributions from the data management community are highlighted with blue italic text. Traditionally, labeling data has been a natural focus of research for machine learning tasks. For example, semi-supervised learning is a classical problem where model training is done on a small amount of labeled data and a larger amount of unlabeled data. However, as machine learning needs to be performed on large amounts of training data, data management issues including how to acquire large datasets, how to perform data labeling at scale, and how to improve the quality of large amounts of existing data become more relevant. Hence, to fully understand the research landscape of data collection, one needs to understand the literature from both the machine learning and data management communities.

    While there are many surveys on data collection that are either limited to one discipline or a class of techniques, to our knowledge, this survey is the first to bridge the machine learning (including natural language processing and computer vision) and data management disciplines. We contend that a machine learning user needs to know the techniques on all sides to make informed decisions on which techniques to use when. In fact, data management plays a role in almost all aspects of machine learning [4], [5]. We note that many sub-topics including semi-supervised learning, active learning, and transfer learning are large enough to have their own surveys. The goal of this survey is not to go into all the depths of these sub-topics, but to focus on breadth and identify what data collection techniques are relevant for machine learning purposes and what research challenges exist. Hence, we will only cover the most representative work of the sub-topics, which are either the best-performing or most recent ones. The key audience of this survey can be researchers or practitioners that are starting to use data collection for machine learning and need an overall landscape introduction. Since the data collection techniques come from different disciplines, some may involve relational data while others non-relational data (e.g., images and text). Sometimes the boundary between operations (e.g., data acquisition and data labeling) is not clear cut. In those cases, we will clarify that the techniques are relevant in multiple operations.

    Motivating Example To motivate the need to explore the techniques in Figure 1, we present a running example on data collection based on our experience with collaborating with the industry on a smart factory application. Suppose that Sally is a data scientist who works on product quality for a smart factory. The factory may produce manufacturing components like gears where it is important for them not to have scratches, dents, or any foreign substance. Sally may want to train a model on images of the components, which can be used to automatically classify whether each product has defects or not. This application scenario is depicted in Figure 3. A general decision flow chart of the data collection techniques that Sally can use is shown in Figure 2. Although the chart may look complicated at first glance, we contend that it is necessary to understand the entire research landscape to make informed decisions for data collection. In comparison, recent commercial tools [6]-[8] only cover a subset of all the possible data collection techniques. When using the chart, one can quickly narrow down the options in two steps by deciding whether to perform one of data acquisition, data labeling, or existing data improvements,

    Fig. 2: A decision flow chart for data collection. From the top left, Sally can start by asking whether she has enough data. The following questions lead to specific techniques that can be used for acquiring data, labeling data, or improving existing data or models. This flow chart does not cover all the details in this survey. For example, data labeling techniques like self learning and crowdsourcing can be performed together as described in Section 3.2.1. Also, some questions (e.g., 'Enough labels for self learning?') are not easy to answer and may require an in-depth understanding of the application and data. There are also techniques specific to the data type (images and text), which we detail in the body of the paper.

    <!-- image -->

    Fig. 3: A running example for data collection. A smart factory may produce various images of product components, which are classified as normal or defective by a convolutional neural network model. Unfortunately, with an application this specific, it is often difficult to find enough data for training the model.

    <!-- image -->

    and then choosing the specific technique to use for each operation. For example, if there is no data, then Sally could generate a dataset by installing camera equipment. Then if she has enough budget for human computation, she can use crowdsourcing platforms like Amazon Mechanical Turk to label the product images for defects. We will discuss more details of the flow chart in the following sections.

    The rest of the paper is organized as follows:

    - We review the data acquisition literature, which can be categorized into data discovery, data augmentation, and data generation. Many of the techniques require scalable solutions and have thus been studied by the data management community (Section 2).
    - We review the data labeling literature and group the techniques into three approaches: utilizing existing labels, using crowdsourcing techniques, and using
    - weak supervision. While data labeling is traditionally a machine learning topic, it is also studied in the data management community as scalability becomes an issue (Section 3).
    - We review techniques for improving existing data or models when acquiring and labeling new data is not the best option. Improving data quality through cleaning is a traditional data management topic where recent techniques are increasingly focusing on machine learning applications (Section 4).
    - We put all the techniques together and provide guidelines on how to decide which data collection techniques to use when (Section 5).
    - Based on the current research landscape, we identify interesting future research challenges (Section 6).

    ## 2 DATA ACQUISITION

    The goal of data acquisition is to find datasets that can be used to train machine learning models. There are largely three approaches in the literature: data discovery, data augmentation, and data generation. Data discovery is necessary when one wants to share or search for new datasets and has become important as more datasets are available on the Web and corporate data lakes [19], [75]. Data augmentation complements data discovery where existing datasets are enhanced by adding more external data. Data generation can be used when there is no available external dataset, but it is possible to generate crowdsourced or synthetic datasets instead. The following sections will cover the three operations in more detail. The individual techniques are classified in Table 1.

    TABLE 1: A classification of data acquisition techniques. Some of the techniques can be used together. For example, data can be generated while augmenting existing data.

    | Task              | Approach          | Techniques                                                                                                          |
    |-------------------|-------------------|---------------------------------------------------------------------------------------------------------------------|
    | Data discovery    | Sharing Searching | Collaborative Analysis [9]-[11] Web [12]-[17] Collaborative and Web [18] Data Lake [19]-[23] Web [24]-[34]          |
    | Data augmentation |                   | Deriving Latent Semantics [35]-[37] Entity Augmentation [30], [31] Data Integration [38]-[44]                       |
    | Data generation   | Crowdsourcing     | Gathering [45]-[54] Processing [49], [50], [55], [56] Generative Adversarial Networks [57]-[62] Policies [63], [64] |

    ## 2.1 Data Discovery

    Data discovery can be viewed as two steps. First, the generated data must be indexed and published for sharing. Many collaborative systems are designed to make this process easy. However, other systems are built without the intention of sharing datasets. For these systems, a post-hoc approach must be used where metadata is generated after the datasets are created, without the help of the dataset owners. Next, someone else can search the datasets for their machine learning tasks. Here the key challenges include how to scale the searching and how to tell whether a dataset is suitable for a given machine learning task. While most of the data discovery literature came from the data management community for data science and data analytics, they are also relevant in a machine learning context. However, another challenge in machine learning is data labeling, which we cover in Section 3.

    ## 2.1.1 Data Sharing

    We study data systems that are designed with dataset sharing in mind. These systems may focus on collaborative analysis, publishing on the Web, or both.

    Collaborative Analysis In an environment where data scientists are collaboratively analyzing different versions of datasets, DataHub [9]-[11] can be used to host, share, combine, and analyze them. There are two components: a dataset version control system inspired by Git (a version control system for code) and a hosted platform on top of it, which provides data search, data cleaning, data integration, and data visualization. A common use case of DataHub is where individuals or teams run machine learning tasks on their own versions of a dataset and later merge with other versions if necessary.

    Web A different approach of sharing datasets is to publish them on the Web. Google Fusion Tables [12]-[14] is a cloudbased service for data management and integration. Fusion Tables enables users to upload structured data (e.g., spreadsheets) and provides tools for visually analyzing, filtering, and aggregating the data. The datasets that are published through Fusion Tables on the Web can be crawled by search engines and show up in search results. The datasets are therefore primarily accessible through Web search. Fusion Tables has been widely used in data journalism for creating interactive maps of data and adding them in articles. In addition, there are many data marketplaces including CKAN [15], Quandl [16], and DataMarket [17] where users can buy and sell datasets or find public datasets.

    Collaborative and Web More recently, we are seeing a merging of collaborative and Web-based systems. For example, Kaggle [18] makes it easy to share datasets on the Web and even host data science competitions for models trained on the datasets. A Kaggle competition host posts a dataset along with a description of the challenge. Participants can then experiment with their techniques and compete with each other. After the deadline passes, a prize is given to the winner of the competition. Kaggle currently has thousands of public datasets and code snippets (called kernels) from competitions. In comparison to DataHub and Fusion Tables, the Kaggle datasets are coupled with competitions and are thus more readily usable for machine learning purposes.

    ## 2.1.2 Data Searching

    While the previous data systems are platforms for sharing datasets, as a next logical step, we now explore systems that are mainly designed for searching datasets. This setting is common within large companies or on the Web.

    Data Lake Data searching systems have become more popular with the advent of data lakes [19], [75] in corporate environments where many datasets are generated internally, but they are not easily discoverable by other teams or individuals within the company. Providing a way to search datasets and analyze them has significant business value because the teams or individuals do not have to make redundant efforts to re-generate the datasets for their machine learning tasks. Most of the recent data lake systems have come from the industry. In many cases, it is not feasible for all the dataset owners to publish datasets through one system. Instead, a post-hoc approach becomes necessary where datasets are

    processed for searching after they are created, and no effort is required on the dataset owner's side.

    As an early solution for data lakes, IBM proposed a system [19] that enables datasets to be curated and then searched. IBM estimates that 70% of the time spent on analytic projects is concerned with discovering, cleaning, and integrating datasets that are scattered among many business applications. Thus, IBM takes the stance of creating, filling, maintaining, and governing the data lake where these processes are collectively called data wrangling . When analyzing data, users do not perform the analytics directly on the data lake, but extract data sets and store them separately. Before this step, the users can do a preliminary exploration of datasets, e.g., visualizing them to determine if the dataset is useful and does not contain anomalies that need further investigation. While supporting data curation in the data lake saves users from processing raw data, it does limit the scalability of how many datasets can be indexed.

    More recently, scalability has become a pressing issue for handling data lakes that consists of most datasets in a large company. Google Data Search (GOODS) [20] is a system that catalogues the metadata of tens of billions of datasets from various storage systems within Google. GOODS infers various metadata including owner information and provenance information (by looking up job logs), analyzes the contents of the datasets, and collects input from users. At the core is a central catalog, which contains the metadata and is indexed for data searching. Due to Google's scale, there are many technical challenges including scaling to the number of datasets, supporting a variety of data formats where the costs for extracting metadata may differ, updating the catalog entries due to the frequent churn of datasets, dealing with uncertainty in metadata discovery, computing dataset importance for search ranking, and recovering dataset semantics that are missing. To find datasets, users can use keywords queries on the GOODS frontend and view profile pages of the datasets that appear in the search results. In addition, users can track the provenance of a dataset to see which datasets were used to create the given dataset and those that rely on it.

    Finally, expressive queries are also important for searching a data lake. While GOODS scales, one downside is that it only supports simple keyword queries. This approach is similar to keyword search in databases [76], [77], but the purpose is to find datasets instead of tuples. The DATA CIVILIZER system [21], [22] complements GOODS by focusing more on the discovery aspect of datasets. Specifically, DATA CIVILIZER consists of a module for building a linkage graph of data. Assuming that datasets have schema, the nodes in the linkage graph are columns of tables while edges are relationships like primary key-foreign key (PKFK) relationships. A data discovery module then supports a rich set of discovery queries on the linkage graph, which can help users more easily discover the relevant datasets. DATARAMAN [23] specializes in extracting structured data from semi-structured log datasets in data lakes automatically by learning patterns. AURUM [78], [79] supports data discovery queries on semantically-linked datasets.

    Web As the Web contains large numbers of structured datasets, there have been significant efforts to automati- cally extract the useful ones [32]-[34]. One of the most successful systems is WebTables [24], [25], which automatically extracts structured data that is published online in the form of HTML tables. For example, WebTables extracts all Wikipedia infoboxes. Initially, about 14.1 billion HTML tables are collected from the Google search web crawl. Then a classifier is applied to determine which tables can be viewed as relational database tables. Each relational table consists of a schema that describes the columns and a set of tuples. In comparison to the above data lake systems, WebTables collects structured data from the Web.

    As Web data tends to be much more diverse than say those in a corporate environment, the table extraction techniques have been extended in multiple ways as well. One direction is to extend table extraction beyond identifying HTML tags by extracting relational data in the form of vertical tables and lists and leveraging knowledge bases [27], [28]. Table searching also evolved where, in addition to keyword searching, row-subset queries, entity-attribute queries, and column search were introduced [29]. Finally, techniques for enhancing the tables [30], [31] were proposed where entities or attribute values are added to make the tables more complete.

    Recently, a service called Google Dataset Search [26] was launched for searching repositories of datasets on the Web. The motivation is that there are thousands of data repositories on the Web that contain millions of datasets that are not easy to search. Dataset Search lets dataset providers describe their datasets using various metadata (e.g., author, publication date, how the data was collected, and terms for using the data) so that they become more searcheable. In comparison to the fully-automatic WebTables, dataset providers may need to do some manual work, but have the opportunity to make their datasets more searcheable. In comparison to GOODS, Dataset Search targets the Web instead of a data lake.

   uv run""")

    # --- TIMING BLOCK ADDED HERE ---
    print("Starting chunking process...")
    start_time = time.time()  # Start Timer

    chunks = chunk_markdown(readme_sample)

    end_time = time.time()  # End Timer
    elapsed_time = end_time - start_time
    print(f"\n⏱️  Chunking completed in {elapsed_time:.6f} seconds")
    # -------------------------------

    for i, c in enumerate(chunks):
        m = c["metadata"]
        content_preview = c["content"].replace("\n", "\\n")[:50]

        print(f"[{i}] {m['content_type'].upper()}")
        print(f"    Header: '{m['section_header']}'")
        print(f"    Content: {content_preview}...")
        print("-" * 40)

    # VERIFICATION
    key_chunk = next(
        (c for c in chunks if "Get OpenAI API Key" in c["metadata"]["section_header"]),
        None,
    )

    if key_chunk and "1. Go to" in key_chunk["content"]:
        print(
            "\n✅ SUCCESS: Numbered lists were PRESERVED as content (not converted to headers)."
        )
    else:
        print("\n❌ FAILURE: Numbered lists are missing or converted incorrectly.")
