Can you evaluate my chunks now:



[

  {

​    "chunk_id": "21406039-e56b-4e75-b0fb-d6d35b3b208d",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 0,

​    "content": "## FAIRNESS AND MACHINE LEARNING\r\nLimitations and Opportunities\r\nSolon Barocas, Moritz Hardt, Arvind Narayanan\r\nhttps://fairmlbook.org/",

​    "section_header": "FAIRNESS AND MACHINE LEARNING",

​    "content_type": "prose",

​    "estimated_tokens": 39,

​    "prev_chunk_id": null,

​    "next_chunk_id": "bfe8b768-9f6b-48ad-b5aa-7e3832421784",

​    "quality_score": 0,

​    "should_use": 0

  },

  {

​    "chunk_id": "bfe8b768-9f6b-48ad-b5aa-7e3832421784",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 1,

​    "content": "## Contents\r\n| Preface                                                                      | v   |\r\n|------------------------------------------------------------------------------|-----|\r\n| Acknowledgments                                                              | x   |\r\n| 1 Introduction                                                               | 1   |\r\n| Demographic disparities . . . . . . . . . . . . . . . . . . . . . . . . .    | 3   |\r\n| The machine learning loop . . . . . . . . . . . . . . . . . . . . . . . .    | 4   |\r\n| The state of society . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 5   |\r\n| The trouble with measurement . . . . . . . . . . . . . . . . . . . . .       | 7   |\r\n| From data to models . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 10  |\r\n| The pitfalls of action . . . . . . . . . . . . . . . . . . . . . . . . . . . | 12  |\r\n| Feedback and feedback loops . . . . . . . . . . . . . . . . . . . . . .      | 13  |\r\n| Getting concrete with a toy example . . . . . . . . . . . . . . . . . .      | 15  |\r\n| Justice beyond fair decision making . . . . . . . . . . . . . . . . . .      | 18  |\r\n| Our outlook: limitations and opportunities . . . . . . . . . . . . . .       | 20  |\r\n| Bibliographic notes and further reading . . . . . . . . . . . . . . . .      | 21  |\r\n| 2 When is automated decision making legitimate?                              | 23  |\r\n| Machine learning is not a replacement for human decision making              | 24  |\r\n| Bureaucracy as a bulwark against arbitrary decision making . . . .           | 25  |\r\n| Three Forms of Automation . . . . . . . . . . . . . . . . . . . . . . .      | 27  |\r\n| Mismatch between target and goal . . . . . . . . . . . . . . . . . . .       | 34  |\r\n| Failing to consider relevant information . . . . . . . . . . . . . . . .     | 35  |\r\n| The limits of induction . . . . . . . . . . . . . . . . . . . . . . . . . .  | 38  |\r\n| A right to accurate predictions? . . . . . . . . . . . . . . . . . . . . .   | 39  |\r\n| Agency, recourse, and culpability . . . . . . . . . . . . . . . . . . . .    | 40  |\r\n| Concluding thoughts . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 43  |\r\n| 3 Classification                                                             | 44  |\r\n| Modeling populations as probability distributions . . . . . . . . . .        | 44  |\r\n| Formalizing classification . . . . . . . . . . . . . . . . . . . . . . . .   | 46  |\r\n| Supervised learning . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 50  |\r\n| Groups in the population . . . . . . . . . . . . . . . . . . . . . . . .     | 52  |\r\n| Statistical non-discrimination criteria . . . . . . . . . . . . . . . . . .  | 54  |\r\n| . . . . . . . . . . . . . .                                                             | Separation . . . . . . . . . . . . . . . . . . . . . . . 56                                         |\r\n|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|\r\n| . . . . . . . . . . . . . .                                                             | . . . . . . . . . . . . . . . . . . . . . . . 60                                                    |\r\n| Sufficiency How to satisfy a non-discrimination                                         | criterion . . . . . . . . . . . . . . . . . 63                                                      |\r\n| Relationships between criteria . . .                                                    | . . . . . . . . . . . . . . . . . . . . . . . 64                                                    |\r\n| Case study: Credit scoring . . . . .                                                    | . . . . . . . . . . . . . . . . . . . . . . 67                                                      |\r\n| Inherent limitations of observational criteria                                          | . . . . . . . . . . . . . . . . . . 71                                                              |\r\n| Chapter notes . . . . . . . . . . . .                                                   | . . . . . . . . . . . . . . . . . . . . . . . 73                                                    |\r\n| 4 Relative notions of fairness                                                          | 76                                                                                                  |\r\n| Systematic relative disadvantage .                                                      | . . . . . . . . . . . . . . . . . . . . . . . 76                                                    |\r\n| Six accounts of the wrongfulness of discrimination                                      | . . . . . . . . . . . . . 78                                                                        |\r\n| Equality of opportunity . . . . .                                                       | . . . . . . . . . . . . . . . . . . . . . . . 81                                                    |\r\n| .                                                                                       |                                                                                                     |\r\n| Tensions between the different views                                                    | . . . . . . . . . . . . . . . . . . . . . . 85                                                      |\r\n| Merit and desert . . . . . . . . . . . The cost of fairness . . . . . . . . .           | . . . . . . . . . . . . . . . . . . . . . . . 88 . . . . . . . . . . . . . . . . . . . . . . .      |\r\n| Connecting                                                                              |                                                                                                     |\r\n| statistical and moral notions of fairness                                               | . . . . . . . . . . . . . 92                                                                        |\r\n| The normative underpinnings of error rate parity . .                                    | . . . . . . . . . . . . 97                                                                          |\r\n| Alternatives for realizing the middle view of equality of opportunity                   | . . . 101                                                                                           |\r\n| Summary . . . . . . . . . . . . . . .                                                   | . . . . . . . . . . . . . . . . . . . . . . . 101                                                   |\r\n| 5 Causality                                                                             | 104 105                                                                                             |\r\n| The limitations of observation . . .                                                    | . . . . . . . . . . . . . . . . . . . . . . .                                                       |\r\n| Causal models . . . . . . . . . . . . . . . . . . . . . . . .                           | . . . . . . . . . . . . . . . . . . . . . . . 107 . . . . . . . . . . . . . . . . . . . . . . . 111 |\r\n| Causal graphs                                                                           |                                                                                                     |\r\n| Interventions and causal effects . .                                                    | . . . . . . . . . . . . . . . . . . . . . . . 113                                                   |\r\n| Confounding . . . . . . . . . . . . . .                                                 | . . . . . . . . . . . . . . . . . . . . . . . 114 . . . . . . . . . . . . . . . . . . . . . . . 117 |\r\n| Graphical discrimination analysis . . . . . . . . .                                     |                                                                                                     |\r\n| Counterfactuals . .                                                                     | . . . . . . . . . . . . . . . . . . . . . . . 121                                                   |\r\n| Counterfactual discrimination analysis                                                  | . . . . . . . . . . . . . . . . . . . . . 127                                                       |\r\n| Validity of causal modeling . . . .                                                     | . . . . . . . . . . . . . . . . . . . . . . . 132                                                   |\r\n| Chapter notes . . . . . . . . . . . .                                                   | . . . . . . . . . . . . . . . . . . . . . . . 137                                                   |",

​    "section_header": "Contents",

​    "content_type": "prose",

​    "estimated_tokens": 1938,

​    "prev_chunk_id": "21406039-e56b-4e75-b0fb-d6d35b3b208d",

​    "next_chunk_id": "ace8904f-97db-4a1e-b0c3-0f84203d8e58",

​    "quality_score": 20,

​    "should_use": 0

  },

  {

​    "chunk_id": "ace8904f-97db-4a1e-b0c3-0f84203d8e58",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 2,

​    "content": "| Chapter notes . . . . . . . . . . . .                                                   | . . . . . . . . . . . . . . . . . . . . . . . 137                                                   |\r\n| 6 Understanding United States anti-discrimination law                                   | 139                                                                                                 |\r\n| History and overview of U.S. anti-discrimination law A few basics of the American legal | . . . . . . . . . . . . 140 system . . . . . . . . . . . . . . . . . . 146                          |\r\n| . . . Concluding thoughts . . . . . . . . .                                             | . . . . . . . . . . . . . . . . . . 155 . . . . . . . . . . . . . . . . . . . . . .                 |\r\n| Limits of the law in curbing discrimination Regulating machine learning .               | 160                                                                                                 |\r\n|                                                                                         | . . . . . . . . . . . . . . . . . . . . . . 169                                                     |\r\n| 7 Testing discrimination in practice Part 1 : Traditional tests for discrimination      | 171 . . . . . . . . . . . . . . . . . . 172                                                         |\r\n| factors in decisions                                                                    | . . . . . . . . . . . . . . . . . .                                                                 |\r\n|                                                                                         | 177                                                                                                 |\r\n| Revealing extraneous                                                                    | .                                                                                                   |\r\n|                                                                          | Testing the impact of decisions and interventions . . . . . . . . . . . . . . 178   |\r\n|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------|\r\n| Purely observational tests . . . . . .                                   | . . . . . . . . . . . . . . . . . . . . . . 179                                     |\r\n| Summary of traditional tests and methods                                 | . . . . . . . . . . . . . . . . . . 182                                             |\r\n| Taste-based and statistical discrimination                               | . . . . . . . . . . . . . . . . . . . 184                                           |\r\n| Studies of decision making processes and organizations . .               | . . . . . . . . 186                                                                 |\r\n| Part 2 : Testing discrimination in algorithmic systems                   | . . . . . . . . . . . 187                                                           |\r\n| Fairness considerations in applications of natural                       | language processing . . 188                                                         |\r\n| Demographic disparities and questionable applications of computer vision | 191                                                                                 |\r\n| Search and recommendation systems: three types of harms                  | . . . . . . . . 192                                                                 |\r\n| Understanding unfairness in ad targeting                                 | . . . . . . . . . . . . . . . . . . . 194                                           |\r\n| Fairness considerations in the design                                    | of online marketplaces . . . . . . . . 196                                          |\r\n| Mechanisms of discrimination . . . .                                     | . . . . . . . . . . . . . . . . . . . . . . 198                                     |\r\n| Fairness criteria in algorithmic audits .                                | . . . . . . . . . . . . . . . . . . . . 199                                         |\r\n| Information flow, fairness, privacy . . .                                | . . . . . . . . . . . . . . . . . . . . 201                                         |\r\n| Comparison of research methods . .                                       | . . . . . . . . . . . . . . . . . . . . . . 202                                     |\r\n| Looking ahead . . . . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 204                                     |\r\n| Chapter notes . . . . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 204                                     |\r\n| 8 A broader view of discrimination                                       | 205                                                                                 |\r\n| Case study: the gender earnings gap on Uber                              | . . . . . . . . . . . . . . . . 205                                                 |\r\n| Three levels of discrimination . . . .                                   | . . . . . . . . . . . . . . . . . . . . . . 209                                     |\r\n| Machine learning and structural discrimination .                         | . . . . . . . . . . . . . . 213                                                     |\r\n| Structural interventions for fair machine learning                       | . . . . . . . . . . . . . . 217                                                     |\r\n| Organizational interventions for fairer decision making                  | . . . . . . . . . . . 221                                                           |\r\n| Chapter notes . . . . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 229                                     |\r\n| Appendix: a deeper look at structural factors                            | . . . . . . . . . . . . . . . . . 230                                               |\r\n| 9 Datasets                                                               | 232                                                                                 |\r\n| A tour of datasets in different domains                                  | . . . . . . . . . . . . . . . . . . . . 233                                         |\r\n| Roles datasets play . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 240                                     |\r\n| Harms associated with data . . . . .                                     | . . . . . . . . . . . . . . . . . . . . . . 250                                     |\r\n| Beyond datasets . . . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 254                                     |\r\n| Summary . . . . . . . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 261                                     |\r\n| Chapter notes . . . . . . . . . . . . .                                  | . . . . . . . . . . . . . . . . . . . . . . 261                                     |",

​    "section_header": "Contents",

​    "content_type": "prose",

​    "estimated_tokens": 1350,

​    "prev_chunk_id": "bfe8b768-9f6b-48ad-b5aa-7e3832421784",

​    "next_chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "quality_score": 20,

​    "should_use": 0

  },

  {

​    "chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 3,

​    "content": "## Preface\r\nA peculiar way of making decisions is characteristic of modern society. Institutions of all kinds, from firms to governments, represent populations as data tables. Rows reference individuals. Columns contain measurements about them. Statistical machinery applied to these tables empowers their owners to mine patterns that fit the aggregate.\r\nThen comes a leap of faith. We have to imagine that unknown outcomes, future or unobserved, in the life trajectory of an individual follow the patterns they have found. We must accept decisions made as if all individuals were going to follow the rule of the aggregate. We must pretend to ourselves that to look into the future is to look into the past. It's a leap of faith that has been the basis of consequential decisions for centuries. Fueled by early successes in insurance pricing and financial risk assessment, statistical decision making of this kind has found its way into nearly all aspects of our lives. What accelerated its adoption in recent years has been the explosive growth of machine learning, often under the name of artificial intelligence.\r\nMachine learning shares long established decision-theoretic foundations with large parts of statistics, economics, and computer science. What machine learning adds is a rapidly growing repertoire of heuristics that find decision rules from sufficiently large datasets. These techniques for fitting huge statistical models on large datasets have led to several impressive technological achievements. Image classification, speech recognition, and natural language processing have all made leaps forward. Although these advances often don't directly relate to specific decision making settings, they shape narratives about the new capabilities of machine learning.\r\nAs useful as machine learning is for some positive applications, it is also used to great effect for tracking, surveillance, and warfare. Commercially its most successful use cases to date are targeted advertising and digital content recommendation, both of questionable value to society. From its roots in World War II era cybernetics and control theory, machine learning has always been political. Advances in artificial intelligence feed into a global industrial military complex, and are funded by it. The success stories told about machine learning also support those who would like to adopt algorithms in domains outside those studied by computer scientists. An opaque marketplace of software vendors renders algorithmic decision making tools for use in law enforcement, criminal justice, education, and social services. In many cases what is marketed and sold\r\nas artificial intelligence are statistical methods that virtually haven't changed in decades.\r\nMany take the leap of faith behind statistical decision making for granted to an extent that it's become difficult to question. Entire disciplines have embraced mathematical models of optimal decision making in their theoretical foundations. Much of economic theory takes optimal decisions as an assumption and an ideal of human behavior. In turn, other disciplines label deviations from mathematical optimality as 'bias' that invites elimination. Volumes of academic papers speak to the evident biases of human decision makers.\r\nIn this book, we take machine learning as a reason to revisit this leap of faith and to interrogate how institutions make decisions about individuals. Institutional decision making has long been formalized via bureaucratic procedures and machine learning shares much in common with it. In many cases, machine learning is adopted to improve and sometimes automate the high-stakes decisions routinely made by institutions. Thus, we do not compare machine learning models to the subjective judgments of individual humans, but instead to institutional decisionmaking. Interrogating machine learning is a way of interrogating institutional decision making in society today and for the foreseeable future.\r\nIf machine learning is our way into studying institutional decision making, fairness is the moral lens through which we examine those decisions. Much of our discussion applies to concrete screening, selection, and allocation scenarios. A typical example is that of an employer accepting or rejecting job applicants. One way to construe fairness in such decision making scenarios is as the absence of discrimination. This perspective is micro insofar as individuals are the unit of analysis. We study how measured characteristics of an individual lead to different outcomes. Individuals are the sociological building block. A population is a collection of individuals. Groups are subsets of the population. A decision maker has the power to accept or reject individuals for an opportunity they seek. Discrimination in this view is about wrongful consideration on the basis of group membership. The problem is as much about what wrongful means as what is on the basis of. Discrimination is also not a general concept. It's domain specific as it relates to opportunities that affect people's lives. It's concerned with socially salient categories that have served as the basis for unjustified and systematically adverse treatment.\r\nThe first chapter after the introduction explores the properties that make automated decision making a matter of significant and unique normative concern. In particular, we situate our exploration of machine learning in a longer history of critical reflection on the perils of bureaucratic decision making and its mechanical application of formalized rules. Before we even turn to questions of discrimination, we first ask what makes automated decision-making legitimate in the first place. In so doing, we isolate the specific properties of machine learning that distinguish it from other forms of automation along a range of normative dimensions.\r\nSince the 1950 s, scholars have developed formal models of discrimination that describe the unequal treatment of multiple different groups in the population by a decision maker. In Chapter 3 , we dive into statistical decision theory, allowing\r\nus to formalize a number of fairness criteria. Statistical fairness criteria express different notions of equality between groups. We boil down the vast space of formal definitions to essentially three different mutually exclusive definitions. Each definition resonates with a different moral intuition. None is sufficient to support conclusive claims of fairness. Nor are these definitions suitable targets to optimize for. Satisfying one of these criteria permits blatantly unfair solutions. Despite their significant limitations, these definitions have been influential in the debate around fairness.\r\nChapter 4 explores the normative underpinnings of objections to systematic differences in the treatment of different groups and inequalities in the outcomes experienced by these groups. We review the many accounts of the wrongfulness of discrimination and show how these relate to various views of what it would mean to provide equality of opportunity. In doing so, we highlight some tensions between competing visions of equality of opportunity-some quite narrow and others quite sweeping-and the various arguments that have been advanced to help settle these conflicts. With this in place, we then explore how common moral intuitions and established moral theories can help us make sense of the formalisms introduced in Chapter 3 , with the goal of giving these definitions greater normative substance.\r\nPresent in both technical and legal scholarship on discrimination is the idea of assigning normative weight to causal relationships. Was group membership the cause of rejection? Would the applicant have been rejected had he been of a different race? Would she have been accepted but for her gender? To understand these kinds of statements and the role that causality plays in discrimination, Chapter 5 of this book is a self-contained introduction to the formal concepts of causality.\r\nFollowing our formal encounter with fairness definitions, both statistical and causal, we turn to the legal dimensions of discrimination in the United States in Chapter 6 . The legal situation neither maps cleanly to the moral foundations nor the formal work, complicating the situation considerably. The two dominant legal doctrines, disparate treatment and disparate impact, appear to create a tension between explicit consideration of group membership and intervening to avoid discrimination.\r\nExtending on both the causal and legal chapters, Chapter 7 goes into detail about the complexities of testing for discrimination in practice through experiments and audits.\r\nStudying discrimination in decision making has been criticized as a narrow perspective on a broader system of injustice for at least two reasons. First, as a notion of discrimination it neglects powerful structural determinants of discrimination, such as laws and policies, infrastructure, and education. Second, it orients the space of intervention towards solutions that reform existing decision making systems, in the case of machine learning typically via updates to an algorithm. As such the perspective can seem to prioritize 'tech fixes' over more powerful structural interventions and alternatives to deploying a machine learning system altogether. Rather than predicting failure to appear in court and punishing defendants for it, for example, perhaps the better intervention is to facilitate access to court ap-\r\npointments by providing transportation and child care. Chapter 8 introduces the reader to this broader perspective and its associated space of interventions from an empirical angle.\r\nRecognizing the importance of a broader social and structural perspective, why should we continue to study the notion of discrimination in decision making? One benefit is that it provides a political and legal strategy to put pressure on individual decision makers. We can bring forward claims of discrimination against a specific person, firm, or institution. We can discuss what interventions exist within reasonable proximity to the decision maker that we therefore expect the decision maker to implement. Some such micro interventions may also be more directly feasible than structural interventions.\r\nTaking on a micro perspective decidedly does not mean to ignore context. In fact, allocation rules that avoid explicit consideration of group membership while creating opportunity for a group likely do so by connecting the allocation rule with external social facts. One prominent example is the 'Texas ten percent rule' that guarantees Texas students who graduated in the top ten percent of their high school class automatic admission to all state-funded universities. The rule wouldn't be effective in promoting racial diversity on public university campuses if high school classes weren't segregated to begin with. This example illustrates that there is no mutual exclusivity between examining specific decision rules in detail and paying attention to broader social context. Rather these go hand in hand.\r\nA consequential point of contact between the broader social world and the machine learning ecosystem are datasets. A full chapter explores the history, significance, and scientific basis of machine learning datasets. Detailed consideration of datasets, the collection and construction of data, as well as the harms associated with data tend to be lacking from machine learning curricula.",

​    "section_header": "Preface",

​    "content_type": "prose",

​    "estimated_tokens": 1979,

​    "prev_chunk_id": "ace8904f-97db-4a1e-b0c3-0f84203d8e58",

​    "next_chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 4,

​    "content": "Fairness remains an active research area that is far from settled. We wrote this book during a time of explosive research activity. Thousands of related papers have appeared in the last five years of writing. Many of them propose fairnesspromoting algorithmic interventions. This text is not a survey of this rapidly evolving area, nor is it a definitive reference. The final chapter, available online, provides an entry point to the emerging research on algorithmic interventions.\r\nThe book has some serious, perhaps obvious, limitations.\r\nLarge parts of our book are specific to the United States. Written by three authors educated and employed at US institutions, the book is based on Western moral tradition, assumes the laws and legal theory of the United States, and references the industrial and political context of the United States throughout. We made no attempt to address this serious limitation within this book. Indeed, it would require an entirely different book to address this limitation.\r\nA second limitation stems from the fact that our primary goal was to develop the moral, normative, and technical foundations necessary to engage with the topic. Due to its focus on foundations, the book will strike some as a step removed from the important experiences of those individuals and communities most seriously wronged and harmed by the use of algorithms. This shortcoming is exacerbated by the fact that the authors of this book lack first-hand experience of the systems of\r\noppression that algorithms are a part of. Consequently, this book is no substitute for the vital work of those activists, journalists, and scholars that have taught us about the dangers of algorithmic decision making in context. We build on these essential contributions in writing this book. We aimed to highlight them throughout, anticipating that we likely fell short in some significant ways.\r\nThe book is neither a wholesale endorsement of algorithmic decision making, nor a broad indictment. In writing this book, we attempt what is likely the least popular position on any topic: a balance. We try to work out where algorithmic decision making has merit, while committing significant attention to its harms and limitations. Some will see our balancing act as a lack of political commitments, a sort of bothsideism.\r\nDespite the urgency of the political situation, our book provides no direct practical guide to fair decisions. As a matter of fact, we wrote this book for the long haul. We're convinced that the debates around algorithmic decision making will persist. Our goal is to strengthen the intellectual foundations of debates to come, which will play out in thousands of specific instances. Anyone hoping to shape this future of algorithmic decision making in society will likely find some worthwhile material in this book.\r\nA few chapters, specifically Chapter 3 on classification and Chapter 5 on causality, require significant mathematical prerequisites, primarily in undergraduate probability and statistics. However, the other chapters we dedicate to much broader audiences. We hope that students in multiple fields will find this book helpful in preparing for research in related areas. The book does not fit neatly into the disciplinary boundaries of any single department. As a result it gives readers an opportunity to go beyond established curricula in their primary discipline.\r\nSince we've started publishing material from this book years ago, instructors have incorporated the material into a variety of courses, both at the undergraduate and graduate level, in different departments. Hundreds of readers have sent us tremendously helpful feedback for which we are deeply grateful.\r\nAnd to those lamenting our slow progress in writing this book, we respond empathetically:\r\nThat's fair.",

​    "section_header": "Preface",

​    "content_type": "prose",

​    "estimated_tokens": 699,

​    "prev_chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "next_chunk_id": "3e9337b8-f2af-4ce7-a99e-6f93e79af208",

​    "quality_score": 80,

​    "should_use": 1

  },

  {

​    "chunk_id": "3e9337b8-f2af-4ce7-a99e-6f93e79af208",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 5,

​    "content": "## Acknowledgments\r\nThis book wouldn't have been possible without the profound contributions of our collaborators and the community at large.\r\nWe are grateful to our students for their active participation in pilot courses at Berkeley, Cornell, and Princeton. Thanks in particular to Claudia Roberts for lecture notes of the Princeton course.\r\nSpecial thanks to Katherine Yen for editorial and technical help with the book. Moritz Hardt is indebted to Cynthia Dwork for introducing him to the topic of this book during a formative internship in 2010 .\r\nWe benefitted from substantial discussions, feedback and comments from Rediet Abebe, Andrew Brunskill, Aylin Caliskan, André Cruz, Frances Ding, Michaela Hardt, Lily Hu, Ben Hutchinson, Shan Jiang, Sayash Kapoor, Lauren Kaplan, Niki Kilbertus, Been Kim, Kathy Kleiman, Issa Kohler-Hausmann, Mihir Kshirsagar, Eric Lawrence, Zachary Lipton, Lydia T. Liu, John Miller, Smitha Milli, Shira Mitchell, Jared Moore, Robert Netzorg, David Parkes, Juan Carlos Perdomo, Eike Willi Petersen, Daniele Regoli, Ofir Reich, Claudia Roberts, Olga Russakovsky, Matthew J. Salganik, Carsten Schwemmer, Ludwig Schmidt, Andrew Selbst, Matthew Sun, Angelina Wang, Christo Wilson, Annette Zimmermann, Tijana Zrnic.\r\nArvind Narayanan is grateful for support from the National Science Foundation under grants IIS1763642 and CHS1704444 .",

​    "section_header": "Acknowledgments",

​    "content_type": "prose",

​    "estimated_tokens": 329,

​    "prev_chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "next_chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "quality_score": 0,

​    "should_use": 0

  },

  {

​    "chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 6,

​    "content": "## 1 Introduction\r\nOur success, happiness, and wellbeing are never fully of our own making. Others' decisions can profoundly affect the course of our lives: whether to admit us to a particular school, offer us a job, or grant us a mortgage. Arbitrary, inconsistent, or faulty decision-making thus raises serious concerns because it risks limiting our ability to achieve the goals that we have set for ourselves and access the opportunities for which we are qualified.\r\nSo how do we ensure that these decisions are made the right way and for the right reasons? While there's much to value in fixed rules, applied consistently, good decisions take available evidence into account. We expect admissions, employment, and lending decisions to rest on factors that are relevant to the outcome of interest.\r\nIdentifying details that are relevant to a decision might happen informally and without much thought: employers might observe that people who study math seem to perform particularly well in the financial industry. But they could test these observations against historical evidence by examining the degree to which one's major correlates with success on the job. This is the traditional work of statistics-and it promises to provide a more reliable basis for decision-making by quantifying how much weight to assign certain details in our determinations.\r\nA body of research has compared the accuracy of statistical models to the judgments of humans, even experts with years of experience. In many head-tohead comparisons on fixed tasks, data-driven decisions are more accurate than those based on intuition or expertise. As one example, in a 2002 study, automated underwriting of loans was both more accurate and less racially disparate. 1 These results have been welcomed as a way to ensure that the high-stakes decisions that shape our life chances are both accurate and fair.\r\nMachine learning promises to bring greater discipline to decision-making because it offers to uncover factors that are relevant to decision-making that humans might overlook, given the complexity or subtlety of the relationships in historical evidence. Rather than starting with some intuition about the relationship between certain factors and an outcome of interest, machine learning lets us defer the question of relevance to the data themselves: which factors-among all that we have observed-bear a statistical relationship to the outcome.\r\nUncovering patterns in historical evidence can be even more powerful than this might seem to suggest. Breakthroughs in computer vision-specifically object\r\nrecognition-reveal just how much pattern-discovery can achieve. In this domain, machine learning has helped to overcome a strange fact of human cognition: while we may be able to effortlessly identify objects in a scene, we are unable to specify the full set of rules that we rely upon to make these determinations. We cannot hand code a program that exhaustively enumerates all the relevant factors that allow us to recognize objects from every possible perspective or in all their potential visual configurations. Machine learning aims to solve this problem by abandoning the attempt to teach a computer through explicit instruction in favor of a process of learning by example. By exposing the computer to many examples of images containing pre-identified objects, we hope the computer will learn the patterns that reliably distinguish different objects from one another and from the environments in which they appear.\r\nThis can feel like a remarkable achievement, not only because computers can now execute complex tasks but also because the rules for deciding what appears in an image seem to emerge from the data themselves.\r\nBut there are serious risks in learning from examples. Learning is not a process of simply committing examples to memory. Instead, it involves generalizing from examples: honing in on those details that are characteristic of (say) cats in general, not just the specific cats that happen to appear in the examples. This is the process of induction: drawing general rules from specific examples-rules that effectively account for past cases, but also apply to future, as yet unseen cases, too. The hope is that we'll figure out how future cases are likely to be similar to past cases, even if they are not exactly the same.\r\nThis means that reliably generalizing from historical examples to future cases requires that we provide the computer with good examples: a sufficiently large number of examples to uncover subtle patterns; a sufficiently diverse set of examples to showcase the many different types of appearances that objects might take; a sufficiently well-annotated set of examples to furnish machine learning with reliable ground truth; and so on. Thus, evidence-based decision-making is only as reliable as the evidence on which it is based, and high quality examples are critically important to machine learning. The fact that machine learning is 'evidence-based' by no means ensures that it will lead to accurate, reliable, or fair decisions.\r\nThis is especially true when using machine learning to model human behavior and characteristics. Our historical examples of the relevant outcomes will almost always reflect historical prejudices against certain social groups, prevailing cultural stereotypes, and existing demographic inequalities. And finding patterns in these data will often mean replicating these very same dynamics.\r\nSomething else is lost in moving to automated, predictive decision making. Human decision makers rarely try to maximize predictive accuracy at all costs; frequently, they might consider factors such as whether the attributes used for prediction are morally relevant. For example, although younger defendants are statistically more likely to re-offend, judges are loath to take this into account in deciding sentence lengths, viewing younger defendants as less morally culpable. This is one reason to be cautious of comparisons seemingly showing the superiority\r\nof statistical decision making. 2 Humans are also unlikely to make decisions that are obviously absurd, but this could happen with automated decision making, perhaps due to erroneous data. These and many other differences between human and automated decision making are reasons why decision making systems that rely on machine learning might be unjust.\r\nWe write this book as machine learning begins to play a role in especially consequential decision-making. In the criminal justice system, as alluded to above, defendants are assigned statistical scores that are intended to predict the risk of committing future crimes, and these scores inform decisions about bail, sentencing, and parole. In the commercial sphere, firms use machine learning to analyze and filter resumes of job applicants. And statistical methods are of course the bread and butter of lending, credit, and insurance underwriting.\r\nWe now begin to survey the risks in these and many other applications of machine learning, and provide a critical review of an emerging set of proposed solutions. We will see how even well-intentioned applications of machine learning might give rise to objectionable results.",

​    "section_header": "1 Introduction",

​    "content_type": "prose",

​    "estimated_tokens": 1301,

​    "prev_chunk_id": "3e9337b8-f2af-4ce7-a99e-6f93e79af208",

​    "next_chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 7,

​    "content": "## Demographic disparities\r\nAmazon uses a data-driven system to determine the neighborhoods in which to offer free same-day delivery. A 2016 investigation found stark disparities in the demographic makeup of these neighborhoods: in many U.S. cities, White residents were more than twice as likely as Black residents to live in one of the qualifying neighborhoods. 3\r\nNow, we don't know the details of how Amazon's system works, and in particular we don't know to what extent it uses machine learning. The same is true of many other systems reported on in the press. Nonetheless, we'll use these as motivating examples when a machine learning system for the task at hand would plausibly show the same behavior.\r\nIn Chapter 3 we'll see how to make our intuition about demographic disparities mathematically precise, and we'll see that there are many possible ways of measuring these inequalities. The pervasiveness of such disparities in machine learning applications is a key concern of this book.\r\nWhen we observe disparities, it doesn't imply that the designer of the system intended for such inequalities to arise. Looking beyond intent, it's important to understand when observed disparities can be considered to be discrimination. In turn, two key questions to ask are whether the disparities are justified and whether they are harmful. These questions rarely have simple answers, but the extensive literature on discrimination in philosophy and sociology can help us reason about them.\r\nTo understand why the racial disparities in Amazon's system might be harmful, we must keep in mind the history of racial prejudice in the United States, its relationship to geographic segregation and disparities, and the perpetuation of those inequalities over time. Amazon argued that its system was justified because\r\nit was designed based on efficiency and cost considerations and that race wasn't an explicit factor. Nonetheless, it has the effect of providing different opportunities to consumers at racially disparate rates. The concern is that this might contribute to the perpetuation of long-lasting cycles of inequality. If, instead, the system had been found to be partial to ZIP codes ending in an odd digit, it would not have triggered a similar outcry.\r\nThe term bias is often used to refer to demographic disparities in algorithmic systems that are objectionable for societal reasons. We'll minimize the use of this sense of the word bias in this book, since different disciplines and communities understand the term differently, and this can lead to confusion. There's a more traditional use of the term bias in statistics and machine learning. Suppose that Amazon's estimates of delivery dates/times were consistently too early by a few hours. This would be a case of statistical bias . A statistical estimator is said to be biased if its expected or average value differs from the true value that it aims to estimate. Statistical bias is a fundamental concept in statistics, and there is a rich set of established techniques for analyzing and avoiding it.\r\nThere are many other measures that quantify desirable statistical properties of a predictor or an estimator, such as precision, recall, and calibration. These are similarly well understood; none of them require any knowledge of social groups and are relatively straightforward to measure. The attention to demographic criteria in statistics and machine learning is a relatively new direction. This reflects a change in how we conceptualize machine learning systems and the responsibilities of those building them. Is our goal to faithfully reflect the data? Or do we have an obligation to question the data, and to design our systems to conform to some notion of equitable behavior, regardless of whether or not that's supported by the data currently available to us? These perspectives are often in tension, and the difference between them will become clearer when we delve into stages of machine learning.",

​    "section_header": "Demographic disparities",

​    "content_type": "prose",

​    "estimated_tokens": 732,

​    "prev_chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "next_chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 8,

​    "content": "## The machine learning loop\r\nLet's study the pipeline of machine learning and understand how demographic disparities propagate through it. This approach lets us glimpse into the black box of machine learning and will prepare us for the more detailed analyses in later chapters. Studying the stages of machine learning is crucial if we want to intervene to minimize disparities.\r\nThe figure below shows the stages of a typical system that produces outputs using machine learning. Like any such diagram, it is a simplification, but it is useful for our purposes.\r\nThe first stage is measurement, which is the process by which the state of the world is reduced to a set of rows, columns, and values in a dataset. It's a messy process, because the real world is messy. The term measurement is misleading, evoking an image of a dispassionate scientist recording what she observes, whereas we'll see that it requires subjective human decisions.\r\nFigure 1 . 1 : The machine learning loop\r\nThe 'learning' in machine learning refers to the next stage, which is to turn that data into a model. A model summarizes the patterns in the training data; it makes generalizations. A model could be trained using supervised learning via an algorithm such as Support Vector Machines, or using unsupervised learning via an algorithm such as k-means clustering. It could take many forms: a hyperplane or a set of regions in n-dimensional space, or a set of distributions. It is typically represented as a set of weights or parameters.\r\nThe next stage is the action we take based on the model's predictions , which are applications of the model to new, unseen inputs. By the way, 'prediction' is another misleading term-while it does sometimes involve trying to predict the future ('is this patient at high risk for cancer?'), sometimes it doesn't ('is this social media account a bot?').\r\nPrediction can take the form of classification (determine whether a piece of email is spam), regression (assigning risk scores to defendants), or information retrieval (finding documents that best match a search query).\r\nThe actions in these three applications might be: depositing the email in the user's inbox or spam folder, deciding whether to set bail for the defendant's pretrial release, and displaying the retrieved search results to the user. They may differ greatly in their significance to the individual, but they have in common that the collective responses of individuals to these decisions alter the state of the world-that is, the underlying patterns that the system aims to model.\r\nSome machine learning systems record feedback from users (how users react to actions) and use them to refine the model. For example, search engines track what users click on as an implicit signal of relevance or quality. Feedback can also occur unintentionally, or even adversarially; these are more problematic, as we'll explore later in this chapter.",

​    "section_header": "The machine learning loop",

​    "content_type": "prose",

​    "estimated_tokens": 579,

​    "prev_chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "next_chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 9,

​    "content": "## The state of society\r\nIn this book, we're concerned with applications of machine learning that involve data about people . In these applications, the available training data will likely encode the demographic disparities that exist in our society. For example, the\r\nFigure 1 . 2 : A sample of occupations in the United States in decreasing order of the percentage of women. The area of the bubble represents the number of workers.\r\nfigure shows the gender breakdown of a sample of occupations in the United States, based on data released by the Bureau of Labor Statistics for the year 2017 .\r\nUnsurprisingly, many occupations have stark gender imbalances. If we're building a machine learning system that screens job candidates, we should be keenly aware that this is the baseline we're starting from. It doesn't necessarily mean that the outputs of our system will be inaccurate or discriminatory, but throughout this chapter we'll see how it complicates things.\r\nWhy do these disparities exist? There are many potentially contributing factors, including a history of explicit discrimination, implicit attitudes and stereotypes about gender, and differences in the distribution of certain characteristics by gender. We'll see that even in the absence of explicit discrimination, stereotypes can be selffulfilling and persist for a long time in society. As we integrate machine learning into decision-making, we should be careful to ensure that ML doesn't become a part of this feedback loop.\r\nWhat about applications that aren't about people? Consider 'Street Bump,' a project by the city of Boston to crowdsource data on potholes. The smartphone app automatically detects potholes using data from the smartphone's sensors and sends the data to the city. Infrastructure seems like a comfortably boring application of data-driven decision-making, far removed from the ethical quandaries we've been discussing. And yet! Kate Crawford points out that the data reflect the patterns of smartphone ownership, which are higher in wealthier parts of the city compared to lower-income areas and areas with large elderly populations. 4 The lesson here is that it's rare for machine learning applications to not be about people. In the case of Street Bump, the data is collected by people, and hence reflects demographic disparities; besides, the reason we're interested in improving infrastructure in the first place is its effect on people's lives.\r\nTo drive home the point that most machine learning applications involve people, we analyzed Kaggle, a well-known platform for data science competitions.\r\nWe focused on the top 30 competitions sorted by prize amount. In 14 of these competitions, we observed that the task is to make decisions about individuals. In most of these cases, there exist societal stereotypes or disparities that may be perpetuated by the application of machine learning. For example, the Automated Essay Scoring 5 task seeks algorithms that attempt to match the scores of human graders of student essays. Students' linguistic choices are signifiers of social group membership, and human graders are known to sometimes have prejudices based on such factors. 6 , 7 Thus, because human graders must provide the original labels, automated grading systems risk enshrining any such discriminatory patterns that are captured in the training data.\r\nIn a further 5 of the 30 competitions, the task did not call for making decisions about people, but decisions made using the model would nevertheless directly impact people. For example, one competition sponsored by real-estate company Zillow calls for improving the company's 'Zestimate' algorithm for predicting home sale prices. Any system that predicts a home's future sale price (and publicizes these predictions) is likely to create a self-fulfilling feedback loop in which homes predicted to have lower sale prices deter future buyers, suppressing demand and lowering the final sale price.\r\nIn 9 of the 30 competitions, we did not find an obvious, direct impact on people, such as a competition on predicting ocean health (of course, even such competitions have indirect impacts on people, due to actions that we might take on the basis of the knowledge gained). In two cases, we didn't have enough information to make a determination.\r\nTo summarize, human society is full of demographic disparities, and training data will likely reflect these. We'll now turn to the process by which training data is constructed, and see that things are even trickier.",

​    "section_header": "The state of society",

​    "content_type": "prose",

​    "estimated_tokens": 872,

​    "prev_chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "next_chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 10,

​    "content": "## The trouble with measurement\r\nThe term measurement suggests a straightforward process, calling to mind a camera objectively recording a scene. In fact, measurement is fraught with subjective decisions and technical difficulties.\r\nConsider a seemingly straightforward task: measuring the demographic diversity of college campuses. A 2017 New York Times article aimed to do just this, and was titled 'Even With Affirmative Action, Blacks and Hispanics Are More Underrepresented at Top Colleges Than 35 Years Ago'. 8 The authors argue that the gap between enrolled Black and Hispanic freshmen and the Black and Hispanic college-age population has grown over the past 35 years. To support their claim, they present demographic information for more than 100 American universities and colleges from the year 1980 to 2015 , and show how the percentages of Black, Hispanic, Asian, White, and multiracial students have changed over the years. Interestingly, the multiracial category was only recently introduced in 2008 , but the comparisons in the article ignore the introduction of this new category. How many students who might have checked the 'White' or 'Black' box checked\r\nthe 'multiracial' box instead? How might this have affected the percentages of 'White' and 'Black' students at these universities? Furthermore, individuals' and society's conception of race changes over time. Would a person with Black and Latino parents be more inclined to self-identify as Black in 2015 than in the 1980 s? The point is that even a seemingly straightforward question about trends in demographic diversity is impossible to answer without making some assumptions, and illustrates the difficulties of measurement in a world that resists falling neatly into a set of checkboxes. Race is not a stable category; how we measure race often changes how we conceive of it, and changing conceptions of race may force us to alter what we measure.\r\nTo be clear, this situation is typical: measuring almost any attribute about people is similarly subjective and challenging. If anything, things are more chaotic when machine learning researchers have to create categories, as is often the case.\r\nOne area where machine learning practitioners often have to define new categories is in defining the target variable. 9 This is the outcome that we're trying to predict - will the defendant recidivate if released on bail? Will the candidate be a good employee if hired? And so on.\r\nBiases in the definition of the target variable are especially critical, because they are guaranteed to bias the predictions relative to the actual construct we intended to predict, as is the case when we use arrests as a measure of crime, or sales as a measure of job performance, or GPA as a measure of academic success. This is not necessarily so with other attributes. But the target variable is arguably the hardest from a measurement standpoint, because it is often a construct that is made up for the purposes of the problem at hand rather than one that is widely understood and measured. For example, 'creditworthiness' is a construct that was created in the context of the problem of how to successfully extend credit to consumers; 9 it is not an intrinsic property that people either possess or lack.\r\nIf our target variable is the idea of a 'good employee', we might use performance review scores to quantify it. This means that our data inherits any biases present in managers' evaluations of their reports. Another example: the use of computer vision to automatically rank people's physical attractiveness. 10 , 11 The training data consists of human evaluation of attractiveness, and, unsurprisingly, all these classifiers showed a preference for lighter skin.\r\nIn some cases we might be able to get closer to a more objective definition for a target variable, at least in principle. For example, in criminal risk assessment, the training data is not judges' decisions about bail, but rather based on who actually went on to commit a crime. But there's at least one big caveat-we can't really measure who committed a crime, so we use arrests as a proxy. This means that the training data contain distortions not due to the prejudices of judges but due to discriminatory policing. On the other hand, if our target variable is whether the defendant appears or fails to appear in court for trial, we would be able to measure it directly with perfect accuracy. That said, we may still have concerns about a system that treats defendants differently based on predicted probability of appearance, given that some reasons for failing to appear are less objectionable than others (trying to hold down a job that would not allow for time off versus\r\ntrying to avoid prosecution). 12\r\nIn hiring, instead of relying on performance reviews for (say) a sales job, we might rely on the number of sales closed. But is that an objective measurement or is it subject to the prejudices of the potential customers (who might respond more positively to certain salespeople than others) and workplace conditions (which might be a hostile environment for some, but not others)?\r\nIn some applications, researchers repurpose an existing scheme of classification to define the target variable rather than creating one from scratch. For example, an object recognition system can be created by training a classifier on ImageNet, a database of images organized in a hierarchy of concepts. 13 ImageNet's hierarchy comes from Wordnet, a database of words, categories, and the relationships among them. 14 Wordnet's authors in turn imported the word lists from a number of older sources, such as thesauri. As a result, WordNet (and ImageNet) categories contain numerous outmoded words and associations, such as occupations that no longer exist and stereotyped gender associations. 15\r\nWe think of technology changing rapidly and society being slow to adapt, but at least in this instance, the categorization scheme at the heart of much of today's machine learning technology has been frozen in time while social norms have changed.\r\nOur favorite example of measurement bias has to do with cameras, which we referenced at the beginning of the section as the exemplar of dispassionate observation and recording. But are they?\r\nThe visual world has an essentially infinite bandwidth compared to what can be captured by cameras, whether film or digital, which means that photography technology involves a series of choices about what is relevant and what isn't, and transformations of the captured data based on those choices. Both film and digital cameras have historically been more adept at photographing lighter-skinned individuals. 16 One reason is the default settings such as color balance which were optimized for lighter skin tones. Another, deeper reason is the limited 'dynamic range' of cameras, which makes it hard to capture brighter and darker tones in the same image. This started changing in the 1970 s, in part due to complaints from furniture companies and chocolate companies about the difficulty of photographically capturing the details of furniture and chocolate respectively! Another impetus came from the increasing diversity of television subjects at this time.\r\nWhen we go from individual images to datasets of images, we introduce another layer of potential biases. Consider the image datasets that are used to train today's computer vision systems for tasks such as object recognition. If these datasets were representative samples of an underlying visual world, we might expect that a computer vision system trained on one such dataset would do well on another dataset. But in reality, we observe a big drop in accuracy when we train and test on different datasets. 17 This shows that these datasets are biased relative to each other in a statistical sense, and is a good starting point for investigating whether these biases include cultural stereotypes.\r\nIt's not all bad news: machine learning can in fact help mitigate measure-\r\nment biases. Returning to the issue of dynamic range in cameras, computational techniques, including machine learning, are making it possible to improve the representation of tones in images. 18 , 19 , 20 Another example comes from medicine: diagnoses and treatments are sometimes personalized by race. But it turns out that race is used as a crude proxy for ancestry and genetics, and sometimes environmental and behavioral factors. 21 , 22 If we can measure the factors that are medically relevant and incorporate them-instead of race-into statistical models of disease and drug response, we can increase the accuracy of diagnoses and treatments while mitigating racial disparities.\r\nTo summarize, measurement involves defining variables of interest, the process for interacting with the real world and turning observations into numbers, and then actually collecting the data. Often machine learning practitioners don't think about these steps, because someone else has already done those things. And yet it is crucial to understand the provenance of the data. Even if someone else has collected the data, it's almost always too messy for algorithms to handle, hence the dreaded 'data cleaning' step. But the messiness of the real world isn't just an annoyance to be dealt with by cleaning. It is a manifestation of a diverse world in which people don't fit neatly into categories. Being inattentive to these nuances can particularly hurt marginalized populations.",

​    "section_header": "The trouble with measurement",

​    "content_type": "prose",

​    "estimated_tokens": 1812,

​    "prev_chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "next_chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 11,

​    "content": "## From data to models\r\nWe've seen that training data reflects the disparities, distortions, and biases from the real world and the measurement process. This leads to an obvious question: when we learn a model from such data, are these disparities preserved, mitigated, or exacerbated?\r\nPredictive models trained with supervised learning methods are often good at calibration: ensuring that the model's prediction subsumes all features in the data for the purpose of predicting the outcome. But calibration also means that by default, we should expect our models to faithfully reflect disparities found in the input data.\r\nHere's another way to think about it. Some patterns in the training data (smoking is associated with cancer) represent knowledge that we wish to mine using machine learning, while other patterns (girls like pink and boys like blue) represent stereotypes that we might wish to avoid learning. But learning algorithms have no general way to distinguish between these two types of patterns, because they are the result of social norms and moral judgments. Absent specific intervention, machine learning will extract stereotypes, including incorrect and harmful ones, in the same way that it extracts knowledge.\r\nA telling example of this comes from machine translation. The screenshot on the right shows the result of translating sentences from English to Turkish and back. 23 The same stereotyped translations result for many pairs of languages and other occupation words in all translation engines we've tested. It's easy to see why. Turkish has gender neutral pronouns, and when translating such a pronoun\r\nFigure 1 . 3 : Translating from English to Turkish, then back to English injects gender stereotypes.\r\nto English, the system picks the sentence that best matches the statistics of the training set (which is typically a large, minimally curated corpus of historical text and text found on the web).\r\nWhen we build a statistical model of language from such text, we should expect the gender associations of occupation words to roughly mirror real-world labor statistics. In addition, because of the male-as-norm bias 24 (the use of male pronouns when the gender is unknown) we should expect translations to favor male pronouns. It turns out that when we repeat the experiment with dozens of occupation words, these two factors-labor statistics and the male-as-norm bias-together almost perfectly predict which pronoun will be returned. 23\r\nHere's a tempting response to the observation that models reflect data biases. Suppose we're building a model for scoring resumes for a programming job. What if we simply withhold gender from the data? Is that a sufficient response to concerns about gender discrimination? Unfortunately, it's not that simple, because of the problem of proxies 9 or redundant encodings, 25 as we'll discuss in Chapter 3 . There are any number of other attributes in the data that might correlate with gender. For example, in our society, the age at which someone starts programming is correlated with gender. This illustrates why we can't just get rid of proxies: they may be genuinely relevant to the decision at hand. How long someone has been programming is a factor that gives us valuable information about their suitability for a programming job, but it also reflects the reality of gender stereotyping.\r\nAnother common reason why machine learning might perform worse for some groups than others is sample size disparity. If we construct our training set by sampling uniformly from the training data, then by definition we'll have fewer data\r\npoints about minorities. Of course, machine learning works better when there's more data, so it will work less well for members of minority groups, assuming that members of the majority and minority groups are systematically different in terms of the prediction task. 25\r\nWorse, in many settings minority groups are underrepresented relative to population statistics. For example, minority groups are underrepresented in the tech industry. Different groups might also adopt technology at different rates, which might skew datasets assembled form social media. If training sets are drawn from these unrepresentative contexts, there will be even fewer training points from minority individuals.\r\nWhen we develop machine-learning models, we typically only test their overall accuracy; so a ' 5 % error' statistic might hide the fact that a model performs terribly for a minority group. Reporting accuracy rates by group will help alert us to problems like the above example. In Chapter 3 , we'll look at metrics that quantify the error-rate disparity between groups.\r\nThere's one application of machine learning where we find especially high error rates for minority groups: anomaly detection. This is the idea of detecting behavior that deviates from the norm as evidence of abuse against a system. A good example is the Nymwars controversy, where Google, Facebook, and other tech companies aimed to block users who used uncommon (hence, presumably fake) names.\r\nFurther, suppose that in some cultures, most people receive names from a small set of names, whereas in other cultures, names might be more diverse, and it might be common for names to be unique. For users in the latter culture, a popular name would be more likely to be fake. In other words, the same feature that constitutes evidence towards a prediction in one group might constitute evidence against the prediction for another group. 25\r\nIf we're not careful, learning algorithms will generalize based on the majority culture, leading to a high error rate for minority groups. Attempting to avoid this by making the model more complex runs into a different problem: overfitting to the training data, that is, picking up patterns that arise due to random noise rather than true differences. One way to avoid this is to explicitly model the differences between groups, although there are both technical and ethical challenges associated with this.",

​    "section_header": "From data to models",

​    "content_type": "prose",

​    "estimated_tokens": 1150,

​    "prev_chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "next_chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 12,

​    "content": "## The pitfalls of action\r\nAny real machine-learning system seeks to make some change in the world. To understand its effects, then, we have to consider it in the context of the larger socio-technical system in which it is embedded.\r\nIn Chapter 3 , we'll see that if a model is calibrated-it faithfully captures the patterns in the underlying data-predictions made using that model will inevitably have disparate error rates for different groups, if those groups have different base rates , that is, rates of positive or negative outcomes. In other words, understanding\r\nthe properties of a prediction requires understanding not just the model, but also the population differences between the groups on which the predictions are applied.\r\nFurther, population characteristics can shift over time; this is a well-known machine learning phenomenon known as drift. If sub-populations change differently over time, but the model isn't retrained, that can introduce disparities. An additional wrinkle: whether or not disparities are objectionable may differ between cultures, and may change over time as social norms evolve.\r\nWhen people are subject to automated decisions, their perception of those decisions depends not only on the outcomes but also the process of decisionmaking. An ethical decision-making process might require, among other things, the ability to explain a prediction or decision, which might not be feasible with black-box models.\r\nA major limitation of machine learning is that it only reveals correlations, but we often use its predictions as if they reveal causation. This is a persistent source of problems. For example, an early machine learning system in healthcare famously learned the seemingly nonsensical rule that patients with asthma had lower risk of developing pneumonia. This was a true pattern in the data, but the likely reason was that asthmatic patients were more likely to receive in-patient care. 26 So it's not valid to use the prediction to decide whether or not to admit a patient. We'll discuss causality in Chapter 5 .\r\nAnother way to view this example is that the prediction affects the outcome (because of the actions taken on the basis of the prediction), and thus invalidates itself. The same principle is also seen in the use of machine learning for predicting traffic congestion: if sufficiently many people choose their routes based on the prediction, then the route predicted to be clear will in fact be congested. The effect can also work in the opposite direction: the prediction might reinforce the outcome, resulting in feedback loops. To better understand how, let's talk about the final stage in our loop: feedback.",

​    "section_header": "The pitfalls of action",

​    "content_type": "prose",

​    "estimated_tokens": 510,

​    "prev_chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "next_chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 13,

​    "content": "## Feedback and feedback loops\r\nMany systems receive feedback when they make predictions. When a search engine serves results, it typically records the links that the user clicks on and how long the user spends on those pages, and treats these as implicit signals about which results were found to be most relevant. When a video sharing website recommends a video, it uses the thumbs up/down feedback as an explicit signal. Such feedback is used to refine the model.\r\nBut feedback is tricky to interpret correctly. If a user clicked on the first link on a page of search results, is that simply because it was first, or because it was in fact the most relevant? This is again a case of the action (the ordering of search results) affecting the outcome (the link(s) the user clicks on). This is an active area of research; there are techniques that aim to learn accurately from this kind of biased feedback. 27\r\nBias in feedback might also reflect cultural prejudices, which is of course much harder to characterize than the effects of the ordering of search results. For example, the clicks on the targeted ads that appear alongside search results might reflect gender and racial stereotypes. There's a well-known study by Latanya Sweeney that hints at this: Google searches for Black-sounding names such as 'Latanya Farrell' were much more likely to results in ads for arrest records ('Latanya Farrell, Arrested?') than searches for White-sounding names ('Kristen Haring'). 28 One potential explanation is that users are more likely to click on ads that conform to stereotypes, and the advertising system is optimized for maximizing clicks.\r\nIn other words, even feedback that's designed into systems can lead to unexpected or undesirable biases. But on top of that, there are many unintended ways in which feedback might arise, and these are more pernicious and harder to control. Let's look at three.\r\nSelf-fulfilling predictions. Suppose a predictive policing system determines certain areas of a city to be at high risk for crime. More police officers might be deployed to such areas. Alternatively, officers in areas predicted to be high risk might be subtly lowering their threshold for stopping, searching, or arresting people-perhaps even unconsciously. Either way, the prediction will appear to be validated, even if it had been made purely based on data biases.\r\nHere's another example of how acting on a prediction can change the outcome. In the United States, some criminal defendants are released prior to trial, whereas for others, a bail amount is set as a precondition of release. Many defendants are unable to post bail. Does the release or detention affect the outcome of the case? Perhaps defendants who are detained face greater pressure to plead guilty. At any rate, how could one possibly test the causal impact of detention without doing an experiment? Intriguingly, we can take advantage of a pseudo-experiment, namely that defendants are assigned bail judges quasi-randomly, and some judges are stricter than others. Thus, pre-trial detention is partially random, in a quantifiable way. Studies using this technique have confirmed that detention indeed causes an increase in the likelihood of a conviction. 29 If bail were set based on risk predictions, whether human or algorithmic, and we evaluated its efficacy by examining case outcomes, we would see a self-fulfilling effect.\r\nPredictions that affect the training set. Continuing this example, predictive policing activity will lead to arrests, records of which might be added to the algorithm's training set. These areas might then continue to appear to be at high risk of crime, and perhaps also other areas with a similar demographic composition, depending on the feature set used for predictions. The disparities might even compound over time.\r\nA 2016 paper by Lum and Isaac analyzed a predictive policing algorithm by PredPol. This is of the few predictive policing algorithms to be published in a peer-reviewed journal, for which the company deserves praise. By applying the algorithm to data derived from Oakland police records, the authors found that Black people would be targeted for predictive policing of drug crimes at roughly twice the rate of White people, even though the two groups have roughly equal rates of drug use. 30 Their simulation showed that this initial bias would be\r\namplified by a feedback loop, with policing increasingly concentrated on targeted areas. This is despite the fact that the PredPol algorithm does not explicitly take demographics into account.\r\nA follow-up paper built on this idea and showed mathematically how feedback loops occur when data discovered on the basis of predictions are used to update the model. 31 The paper also shows how to tweak the model to avoid feedback loops in a simulated setting: by quantifying how surprising an observation of crime is given the predictions, and only updating the model in response to surprising events.\r\nPredictions that affect the phenomenon and society at large. Prejudicial policing on a large scale, algorithmic or not, will affect society over time, contributing to the cycle of poverty and crime. This is a well-trodden thesis, and we'll briefly review the sociological literature on durable inequality and the persistence of stereotypes in Chapter 8 .\r\nLet us remind ourselves that we deploy machine learning so that we can act on its predictions. It is hard to even conceptually eliminate the effects of predictions on outcomes, future training sets, the phenomena themselves, or society at large. The more central machine learning becomes in our lives, the stronger this effect.\r\nReturning to the example of a search engine, in the short term it might be possible to extract an unbiased signal from user clicks, but in the long run, results that are returned more often will be linked to and thus rank more highly. As a side effect of fulfilling its purpose of retrieving relevant information, a search engine will necessarily change the very thing that it aims to measure, sort, and rank. Similarly, most machine learning systems will affect the phenomena that they predict. This is why we've depicted the machine learning process as a loop.\r\nThroughout this book we'll learn methods for mitigating societal biases in machine learning, but we should keep in mind that there are fundamental limits to what we can achieve, especially when we consider machine learning as a socio-technical system instead of a mathematical abstraction. The textbook model of training and test data being independent and identically distributed is a simplification, and might be unachievable in practice.",

​    "section_header": "Feedback and feedback loops",

​    "content_type": "prose",

​    "estimated_tokens": 1290,

​    "prev_chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "next_chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 14,

​    "content": "## Getting concrete with a toy example\r\nNow let's look at a concrete setting, albeit a toy problem, to illustrate many of the ideas discussed so far, and some new ones.\r\nLet's say you're on a hiring committee, making decisions based on just two attributes of each applicant: their college GPA and their interview score (we did say it's a toy problem!). We formulate this as a machine-learning problem: the task is to use these two variables to predict some measure of the 'quality' of an applicant. For example, it could be based on the average performance review score after two years at the company. We'll assume we have data from past candidates that allows us to train a model to predict performance scores based on GPA and interview score.\r\nFigure 1 . 4 : Toy example: a hiring classifier that predicts job performance (not shown) based on GPA and interview score, and then applies a cutoff.\r\nObviously, this is a reductive formulation-we're assuming that an applicant's worth can be reduced to a single number, and that we know how to measure that number. This is a valid criticism, and applies to most applications of data-driven decision-making today. But it has one big advantage: once we do formulate the decision as a prediction problem, statistical methods tend to do better than humans, even domain experts with years of training, in making decisions based on noisy predictors.\r\nGiven this formulation, the simplest thing we can do is to use linear regression to predict the average job performance rating from the two observed variables, and then use a cutoff based on the number of candidates we want to hire. The figure above shows what this might look like. In reality, the variables under consideration need not satisfy a linear relationship, thus suggesting the use of a non-linear model, which we avoid for simplicity.\r\nAs you can see in the figure, our candidates fall into two demographic groups, represented by triangles and squares. This binary categorization is a simplification for the purposes of our thought experiment. But when building real systems, enforcing rigid categories of people can be ethically questionable.\r\nNote that the classifier didn't take into account which group a candidate belonged to. Does this mean that the classifier is fair? We might hope that it is, based on the fairness-as-blindness idea, symbolized by the icon of Lady Justice wearing a blindfold. In this view, an impartial model-one that doesn't use the group membership in the regression-is fair; a model that gives different scores to otherwise-identical members of different groups is discriminatory.\r\nWe'll defer a richer understanding of what fairness means to later chapters, so let's ask a simpler question: are candidates from the two groups equally likely to be positively classified? The answer is no: the triangles are more likely to be\r\nselected than the squares. That's because data is a social mirror; the 'ground truth' labels that we're predicting-job performance ratings-are systematically lower for the squares than the triangles.\r\nThere are many possible reasons for this disparity. First, the managers who score the employees' performance might discriminate against one group. Or the overall workplace might be less welcoming one group, preventing them from reaching their potential and leading to lower performance. Alternately, the disparity might originate before the candidates were hired. For example, it might arise from disparities in educational institutions attended by the two groups. Or there might be intrinsic differences between them. Of course, it might be a combination of these factors. We can't tell from our data how much of the disparity is attributable to these different factors. In general, such a determination is methodologically hard, and requires causal reasoning. 32\r\nFor now, let's assume that we have evidence that the level of demographic disparity produced by our selection procedure is unjustified, and we're interested in intervening to decrease it. How could we do it? We observe that GPA is correlated with the demographic attribute-it's a proxy. Perhaps we could simply omit that variable as a predictor? Unfortunately, we'd also hobble the accuracy of our model. In real datasets, most attributes tend to be proxies for demographic variables, and dropping them may not be a reasonable option.\r\nAnother crude approach is to pick different cutoffs so that candidates from both groups have the same probability of being hired. Or we could mitigate the demographic disparity instead of eliminating it, by decreasing the difference in the cutoffs.\r\nGiven the available data, there is no mathematically principled way to know which cutoffs to pick. In some situations there is a legal baseline: for example, guidelines from the U.S. Equal Employment Opportunity Commission state that if the probability of selection for two groups differs by more than 20 %, it might constitute a sufficient disparate impact to initiate a lawsuit. But a disparate impact alone is not illegal; the disparity needs to be unjustified or avoidable for courts to find liability. Even these quantitative guidelines do not provide easy answers or bright lines.\r\nAt any rate, the pick-different-thresholds approach to mitigating disparities seems unsatisfying, because it is crude and uses the group attribute as the sole criterion for redistribution. It does not account for the underlying reasons why two candidates with the same observable attributes (except for group membership) may be deserving of different treatment.\r\nBut there are other possible interventions, and we'll discuss one. To motivate it, let's take a step back and ask why the company wants to decrease the demographic disparity in hiring.\r\nOne answer is rooted in justice to individuals and the specific social groups to which they belong. But a different answer comes from the firm's selfish interests: diverse teams work better. 33 , 34 From this perspective, increasing the diversity of the cohort that is hired would benefit the firm and everyone in the cohort. As an analogy, picking 11 goalkeepers, even if individually excellent, would make for a\r\npoor soccer team.\r\nHow do we operationalize diversity in a selection task? If we had a distance function between pairs of candidates, we could measure the average distance between selected candidates. As a strawman, let's say we use the Euclidean distance based on the GPA and interview score. If we incorporated such a diversity criterion into the objective function, it would result in a model where the GPA is weighted less. This technique doesn't explicitly consider the group membership. Rather, as a side-effect of insisting on diversity of the other observable attributes, it also improves demographic diversity. However, a careless application of such an intervention can easily go wrong: for example, the model might give weight to attributes that are completely irrelevant to the task.\r\nMore generally, there are many possible algorithmic interventions beyond picking different thresholds for different groups. In particular, the idea of a similarity function between pairs of individuals is a powerful one, and we'll see other interventions that make use of it. But coming up with a suitable similarity function in practice isn't easy: it may not be clear which attributes are relevant, how to weight them, and how to deal with correlations between attributes.",

​    "section_header": "Getting concrete with a toy example",

​    "content_type": "prose",

​    "estimated_tokens": 1431,

​    "prev_chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "next_chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 15,

​    "content": "## Justice beyond fair decision making\r\nThe core concern of this book is group disparities in decision making. But ethical obligations don't end with addressing those disparities. Fairly rendered decisions under unfair circumstances may do little to improve people's lives. In many cases, we cannot achieve any reasonable notion of fairness through changes to decisionmaking alone; we need to change the conditions under which these decisions are made. In other cases, the very purpose of the system might be oppressive, and we should ask whether it should be deployed at all.\r\nFurther, decision making systems aren't the only places where machine learning is used that can harm people: for example, online search and recommendation algorithms are also of concern, even though they don't make decisions about people. Let's briefly discuss these broader questions.",

​    "section_header": "Justice beyond fair decision making",

​    "content_type": "prose",

​    "estimated_tokens": 157,

​    "prev_chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "next_chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "quality_score": 95,

​    "should_use": 1

  },

  {

​    "chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 16,

​    "content": "## Interventions that target underlying inequities\r\nLet's return to the hiring example above. When using machine learning to make predictions about how someone might fare in a specific workplace or occupation, we tend to treat the environment that people will confront in these roles as a constant and ask how people's performance will vary according to their observable characteristics. In other words, we treat the current state of the world as a given, leaving us to select the person who will do best under these circumstances. This approach risks overlooking more fundamental changes that we could make to the workplace (culture, family friendly policies, on-the-job training) that might make it a more welcoming and productive environment for people that have not flourished under previous conditions. 35\r\nThe tendency with work on fairness in machine learning is to ask whether an employer is using a fair selection process, even though we might have the opportunity to intervene in the workplace dynamics that actually account for differences in predicted outcomes along the lines of race, gender, disability, and other characteristics. 36\r\nWe can learn a lot from the so-called social model of disability, which views a predicted difference in a disabled person's ability to excel on the job as the result of a lack of appropriate accommodations (an accessible workplace, necessary equipment, flexible working arrangements) rather than any inherent capacity of the person. A person is only disabled in the sense that we have not built physical environments or adopted appropriate policies to ensure their equal participation.\r\nThe same might be true of people with other characteristics, and changes to the selection process alone will not help us address the fundamental injustice of conditions that keep certain people from contributing as effectively as others. We examine these questions in Chapter 8 .\r\nIt may not be ethical to deploy an automated decision-making system at all if the underlying conditions are unjust and the automated system would only serve to reify it. Or a system may be ill-conceived, and its intended purpose may be unjust, even if it were to work flawlessly and perform equally well for everyone. The question of which automated systems should be deployed shouldn't be left to the logic (and whims) of the marketplace. For example, we may want to regulate the police's access to facial recognition. Our civil rights-freedom or movement and association-are threatened by these technologies both when they fail and when they work well. These are concerns about the legitimacy of an automated decision making system, and we explore them in Chapter 2 .",

​    "section_header": "Interventions that target underlying inequities",

​    "content_type": "prose",

​    "estimated_tokens": 496,

​    "prev_chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "next_chunk_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a",

​    "quality_score": 70,

​    "should_use": 1

  },

  {

​    "chunk_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a",

​    "file_id": "c84c735d-9f17-4d42-adcf-c64bf8b27ef3",

​    "chunk_index": 17,

​    "content": "## The harms of information systems\r\nWhen a defendant is unjustly detailed pre-trial, the harm is clear. But beyond algorithmic decision making, information systems such as search and recommendation algorithms can also have negative effects, but here the harm is indirect and harder to define.\r\nHere's one example. Image search results for occupation terms such as CEO or software developer reflect (and arguably exaggerate) the prevailing gender composition and stereotypes about those occupations. 37 Another example that we encountered earlier is the gender stereotyping in online translation. These and other examples that are disturbing to varying degrees-such as Google's app labeling photos of Black Americans as 'gorillas', or offensive results in autocomplete-seem to fall into a different moral category than, say, a discriminatory system used in criminal justice, which has immediate and tangible consequences.\r\nA talk by Kate Crawford lays out the differences. 38 When decision-making systems in criminal justice, health care, etc. are discriminatory, they create allocative harms , which are caused when a system withholds certain groups an opportunity or a resource. In contrast, the other examples-stereotype perpetuation and cultural denigration-are examples of representational harms , which occur when systems\r\nreinforce the subordination of some groups along the lines of identity-race, class, gender, etc.\r\nAllocative harms have received much attention both because their effects are immediate, and because they are easier to formalize and study in computer science and in economics. Representational harms have long-term effects, and resist formal characterization. But as machine learning has become a part of how we make sense of the world-through technologies such as search, translation, voice assistants, and image labeling-representational harms will leave an imprint on our culture, and influence identity formation and stereotype perpetuation. Thus, these are critical concerns for the fields of natural language processing and computer vision. Although this book is primarily about allocative harms, we will briefly representational harms in Chapters 7 and 9 .\r\nThe majority of content consumed online is mediated by recommendation algorithms that influence which users see which content. Thus, these algorithms influence which messages are amplified. Social media algorithms have been blamed for a litany of ills: echo chambers in which users are exposed to content that conforms to their prior beliefs; exacerbating political polarization; radicalization of some users into fringe beliefs; stoking ethnic resentment and violence; a deterioration of mental health; and so on. Research on these questions is nascent and establishing causality is hard, and it remains unclear how much of these effects are due to the design of the algorithm versus user behavior. But there is little doubt that algorithms have some role. Twitter experimentally compared a non-algorithmic (reverse chronological) content feed to an algorithmic feed, and found that content from the mainstream political right was consistently favored in the algorithmic setting than content from the mainstream political left. 39 While important, this topic is out of scope for us. However, we briefly touch on discrimination in ad targeting and in online marketplaces in Chapter 7 .",

​    "section_header": "The harms of information systems",

​    "content_type": "prose",

​    "estimated_tokens": 621,

​    "prev_chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "next_chunk_id": "51173e3e-513d-42c7-98b2-16d310a9726e",

​    "quality_score": 70,

​    "should_use": 1

  }

]





[

  {

​    "question_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd_q1",

​    "chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "question_text": "What is machine learning?",

​    "answer_text": "Machine learning is a subset of artificial intelligence that involves algorithms that can learn from and make decisions based on data.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd_q2",

​    "chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "question_text": "How does machine learning work?",

​    "answer_text": "Machine learning works by analyzing large datasets to identify patterns and make predictions.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd_q3",

​    "chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "question_text": "What is the significance of bias in decision making?",

​    "answer_text": "Bias is significant because it can lead to unfair outcomes, especially in high-stakes decision making, and it can be hard to detect and correct.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "c7df4608-f832-486a-889f-8da6de6d391e_q1",

​    "chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "question_text": "What is bias?",

​    "answer_text": "Bias is a form of unfairness or discrimination.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "c7df4608-f832-486a-889f-8da6de6d391e_q2",

​    "chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "question_text": "How does decision making use machine learning?",

​    "answer_text": "Decision making uses machine learning to analyze data and make predictions or decisions.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "c7df4608-f832-486a-889f-8da6de6d391e_q3",

​    "chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "question_text": "What is the significance of algorithmic interventions?",

​    "answer_text": "Algorithmic interventions are significant because they can either promote fairness or perpetuate bias, depending on how they are designed and implemented.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6_q1",

​    "chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "question_text": "What is data-driven methods?",

​    "answer_text": "Data-driven methods are decision-making techniques that rely on available evidence and statistical models.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6_q2",

​    "chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "question_text": "How does machine learning enable data-driven methods?",

​    "answer_text": "Machine learning enables data-driven methods by uncovering factors that are relevant to decision-making that humans might overlook, given the complexity or subtlety of the relationships in historical evidence.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6_q3",

​    "chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "question_text": "What is the significance of data-driven methods?",

​    "answer_text": "The significance of data-driven methods is that they improve decision accuracy and fairness by providing a more reliable basis for decision-making by quantifying how much weight to assign certain details in our determinations.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512_q1",

​    "chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "question_text": "What is the system used by Amazon for determining neighborhoods?",

​    "answer_text": "A data-driven system",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512_q2",

​    "chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "question_text": "How does the system used by Amazon for determining neighborhoods work?",

​    "answer_text": "The system uses machine learning to analyze demographic data and optimize delivery times",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512_q3",

​    "chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "question_text": "What is the significance of the racial disparities in Amazon's system?",

​    "answer_text": "The disparities can contribute to the perpetuation of long-lasting cycles of inequality, as they may affect consumer opportunities at racially disparate rates",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "5ffcf616-aba2-4614-b280-77e427b1633a_q1",

​    "chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "question_text": "What is measurement?",

​    "answer_text": "It is the process by which the state of the world is reduced to a set of rows, columns, and values in a dataset.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "5ffcf616-aba2-4614-b280-77e427b1633a_q2",

​    "chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "question_text": "How does learning work?",

​    "answer_text": "Learning involves turning data into a model that summarizes patterns and makes generalizations.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "5ffcf616-aba2-4614-b280-77e427b1633a_q3",

​    "chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "question_text": "What is the significance of demographic disparities?",

​    "answer_text": "Demographic disparities can propagate through the machine learning pipeline, leading to biased outcomes and decisions.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e_q1",

​    "chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "question_text": "What is demographic disparities?",

​    "answer_text": "It refers to the differences in characteristics such as gender, race, and income that exist in a society.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e_q2",

​    "chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "question_text": "How does the data about people used in machine learning applications reflect societal disparities?",

​    "answer_text": "The data reflects the patterns of demographic characteristics that exist in society, such as gender, race, and income.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e_q3",

​    "chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "question_text": "What is the significance of the data about people used in machine learning applications?",

​    "answer_text": "The data reflects the patterns of demographic characteristics that exist in society, such as gender, race, and income, and can perpetuate societal disparities.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "7abf59df-3380-469d-9e79-a74992f2a54f_q1",

​    "chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "question_text": "What is demographic information?",

​    "answer_text": "Demographic information includes data on race, ethnicity, age, gender, and other characteristics of a population.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "7abf59df-3380-469d-9e79-a74992f2a54f_q2",

​    "chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "question_text": "How does demographic information work?",

​    "answer_text": "Demographic information is used to measure and analyze the characteristics of a population, such as race and ethnicity.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "7abf59df-3380-469d-9e79-a74992f2a54f_q3",

​    "chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "question_text": "What is the significance of the target variable?",

​    "answer_text": "The target variable is a construct that is made up for the purposes of the problem at hand rather than one that is widely understood and measured. It is significant because it is often a subjective and challenging attribute to measure.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "26e2b131-5905-4035-b211-817e167b2545_q1",

​    "chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "question_text": "What is a stereotype?",

​    "answer_text": "A stereotype is a widely held but oversimplified notion about a group of people.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "26e2b131-5905-4035-b211-817e167b2545_q2",

​    "chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "question_text": "How does machine learning work?",

​    "answer_text": "Machine learning algorithms learn patterns from data, adjusting parameters to minimize error.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "26e2b131-5905-4035-b211-817e167b2545_q3",

​    "chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "question_text": "What is the significance of sample size disparity?",

​    "answer_text": "It can lead to models that perform worse for minority groups, as they have fewer data points about them.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "926d51d0-77d9-45f8-8d69-6390c307a628_q1",

​    "chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "question_text": "What is disparate error rates?",

​    "answer_text": "Disparate error rates refer to different prediction error rates for different groups in a machine learning system.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "926d51d0-77d9-45f8-8d69-6390c307a628_q2",

​    "chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "question_text": "How does population drift work?",

​    "answer_text": "Population drift occurs when sub-populations change differently over time, but the model isn't retrained, leading to disparities.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "926d51d0-77d9-45f8-8d69-6390c307a628_q3",

​    "chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "question_text": "What is the significance of feedback loops?",

​    "answer_text": "Feedback loops can create disparities in outcomes, and understanding them is crucial for ethical decision-making in machine learning.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5_q1",

​    "chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "question_text": "What is feedback?",

​    "answer_text": "Feedback is the information a system receives about the effectiveness of its actions.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5_q2",

​    "chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "question_text": "How does feedback work?",

​    "answer_text": "Feedback works by providing the system with information about the outcomes of its actions.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5_q3",

​    "chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "question_text": "What is the significance of feedback?",

​    "answer_text": "The significance of feedback is that it can lead to unintended biases and can affect the training set of algorithms, potentially leading to self-fulfilling predictions.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef_q1",

​    "chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "question_text": "What is GPA?",

​    "answer_text": "GPA stands for Grade Point Average, a measure of academic performance.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef_q2",

​    "chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "question_text": "How does GPA work?",

​    "answer_text": "GPA is calculated by summing the grades of all courses taken and dividing by the number of courses. It is used to evaluate academic performance.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef_q3",

​    "chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "question_text": "What is the significance of GPA?",

​    "answer_text": "GPA is significant because it is used to evaluate academic performance and is often used as a factor in hiring decisions.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "4edab167-145b-45cb-b44c-8c322016e721_q1",

​    "chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "question_text": "What is ethical obligations?",

​    "answer_text": "Ethical obligations are the moral duties that people have to others.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "4edab167-145b-45cb-b44c-8c322016e721_q2",

​    "chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "question_text": "How does changes to decisionmaking alone lead to improvements in people's lives?",

​    "answer_text": "Changes to decisionmaking alone may not lead to reasonable notions of fairness, so we need to change the conditions under which these decisions are made.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "4edab167-145b-45cb-b44c-8c322016e721_q3",

​    "chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "question_text": "What is the significance of ethical obligations?",

​    "answer_text": "Ethical obligations are significant because they address the root causes of disparities in decision-making, which can have far-reaching consequences for people's lives.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b_q1",

​    "chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "question_text": "What is a fair selection process?",

​    "answer_text": "A fair selection process is one that treats all candidates equally and does not discriminate based on observable characteristics.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b_q2",

​    "chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "question_text": "How does a fair selection process work?",

​    "answer_text": "A fair selection process uses observable characteristics to predict job performance and selects the candidate who is most likely to succeed in the role.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b_q3",

​    "chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "question_text": "What is the significance of changes to the workplace environment?",

​    "answer_text": "Changes to the workplace environment can make it more welcoming and productive for people, especially those who have not flourished under previous conditions, and can help address fundamental injustices.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  },

  {

​    "question_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a_q1",

​    "chunk_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a",

​    "question_text": "What is allocative harm?",

​    "answer_text": "Allocative harm is caused when a system withholds certain groups an opportunity or a resource.",

​    "difficulty": "Easy",

​    "question_type": "Fact"

  },

  {

​    "question_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a_q2",

​    "chunk_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a",

​    "question_text": "How does algorithmic bias work?",

​    "answer_text": "Algorithmic bias works by system withholding certain groups an opportunity or a resource.",

​    "difficulty": "Medium",

​    "question_type": "Mechanism"

  },

  {

​    "question_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a_q3",

​    "chunk_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a",

​    "question_text": "What is the significance of algorithmic bias?",

​    "answer_text": "The significance of algorithmic bias is that it creates allocative harms, which are caused when a system withholds certain groups an opportunity or a resource.",

​    "difficulty": "Hard",

​    "question_type": "Critical"

  }

]



[

  {

​    "chunk_id": "5c3f787d-d23e-48ae-931b-a7b221c8c0bd",

​    "tags": "[\"machine learning\", \"decision making\", \"institutional\", \"bias\"]",

​    "triplets": "[{\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"decision making\"}, {\"subject\": \"institutional decision making\", \"predicate\": \"is a\", \"object\": \"decision making\"}, {\"subject\": \"bias\", \"predicate\": \"is defined as\", \"object\": \"bias\"}]",

​    "summary": "The text describes the use of machine learning to make decisions about individuals and institutions, highlighting both its positive applications and its potential for misuse.",

​    "processed_at": "2026-01-21T07:19:43.913116"

  },

  {

​    "chunk_id": "c7df4608-f832-486a-889f-8da6de6d391e",

​    "tags": "[\"bias\", \"decision making\", \"institutional\", \"machine learning\"]",

​    "triplets": "[{\"subject\": \"bias\", \"predicate\": \"uses\", \"object\": \"algorithmic interventions\"}, {\"subject\": \"decision making\", \"predicate\": \"uses\", \"object\": \"machine learning\"}, {\"subject\": \"institutional\", \"predicate\": \"uses\", \"object\": \"algorithmic interventions\"}]",

​    "summary": "The text describes the use of machine learning to make decisions about individuals and institutions, highlighting both its positive applications and its potential for misuse.",

​    "processed_at": "2026-01-21T07:20:45.315931"

  },

  {

​    "chunk_id": "1e2b4a34-38fa-4cef-b467-b1515153f7d6",

​    "tags": "[\"data-driven\", \"machine learning\", \"decision-making\", \"accuracy\", \"fairness\"]",

​    "triplets": "[{\"subject\": \"data-driven methods\", \"predicate\": \"uses\", \"object\": \"machine learning\"}, {\"subject\": \"machine learning\", \"predicate\": \"enables\", \"object\": \"data-driven methods\"}, {\"subject\": \"admissions\", \"predicate\": \"uses\", \"object\": \"data-driven methods\"}, {\"subject\": \"employment\", \"predicate\": \"uses\", \"object\": \"data-driven methods\"}, {\"subject\": \"lending\", \"predicate\": \"uses\", \"object\": \"data-driven methods\"}]",

​    "summary": "The text describes the importance of using data-driven methods to make decisions, especially in high-stakes areas like admissions, employment, and lending. It highlights the benefits of machine learning in uncovering relevant factors and improving decision accuracy and fairness.",

​    "processed_at": "2026-01-21T07:21:56.989677"

  },

  {

​    "chunk_id": "53648c9d-0b4c-4596-bdea-f4c20cf78512",

​    "tags": "[\"demographic disparities\", \"machine learning\", \"Zip code\", \"efficiency\", \"cost\"]",

​    "triplets": "[{\"subject\": \"Amazon\", \"predicate\": \"uses\", \"object\": \"data-driven system\"}, {\"subject\": \"data-driven system\", \"predicate\": \"determines\", \"object\": \"neighborhoods for free same-day delivery\"}, {\"subject\": \"neighborhoods\", \"predicate\": \"qualify\", \"object\": \"for free same-day delivery\"}]",

​    "summary": "The text describes the use of machine learning systems to determine neighborhoods for free same-day delivery by Amazon, highlighting racial disparities and the need to understand and mitigate these inequalities.",

​    "processed_at": "2026-01-21T07:23:03.519093"

  },

  {

​    "chunk_id": "5ffcf616-aba2-4614-b280-77e427b1633a",

​    "tags": "[\"measurement\", \"learning\", \"action\", \"demographic disparities\"]",

​    "triplets": "[{\"subject\": \"measurement\", \"predicate\": \"uses\", \"object\": \"dataset\"}, {\"subject\": \"learning\", \"predicate\": \"uses\", \"object\": \"model\"}, {\"subject\": \"action\", \"predicate\": \"uses\", \"object\": \"model\"}, {\"subject\": \"model\", \"predicate\": \"has_property\", \"object\": \"weights or parameters\"}, {\"subject\": \"feedback\", \"predicate\": \"uses\", \"object\": \"model\"}]",

​    "summary": "The text describes the stages of a typical machine learning pipeline, including measurement, learning, and action, and how these stages can lead to demographic disparities.",

​    "processed_at": "2026-01-21T07:24:03.094327"

  },

  {

​    "chunk_id": "9337cf8b-808b-4088-bfa0-8f941d766f6e",

​    "tags": "[\"demographic disparities\", \"gender imbalances\", \"potholes\", \"home sale prices\", \"Zillow\"]",

​    "triplets": "[{\"subject\": \"demographic disparities\", \"predicate\": \"encode\", \"object\": \"data about people\"}]",

​    "summary": "The text describes the ethical considerations and practical applications of machine learning in society, focusing on how data about people can encode societal disparities and how these can be perpetuated by machine learning systems.",

​    "processed_at": "2026-01-21T07:25:09.083675"

  },

  {

​    "chunk_id": "7abf59df-3380-469d-9e79-a74992f2a54f",

​    "tags": "[\"measurement\", \"demographic diversity\", \"race\", \"target variable\"]",

​    "triplets": "[{\"subject\": \"measurement\", \"predicate\": \"uses\", \"object\": \"demographic information\"}, {\"subject\": \"demographic information\", \"predicate\": \"contains\", \"object\": \"race data\"}, {\"subject\": \"race data\", \"predicate\": \"contrasts_with\", \"object\": \"target variable\"}, {\"subject\": \"target variable\", \"predicate\": \"is_a\", \"object\": \"construct\"}, {\"subject\": \"target variable\", \"predicate\": \"defined_as\", \"object\": \"concept\"}]",

​    "summary": "The text describes the challenges and complexities involved in measuring attributes about people, particularly race and demographic diversity.",

​    "processed_at": "2026-01-21T07:26:25.437832"

  },

  {

​    "chunk_id": "26e2b131-5905-4035-b211-817e167b2545",

​    "tags": "[\"disparities\", \"gender\", \"occupation\", \"machine learning\", \"training data\"]",

​    "triplets": "[{\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"data\"}, {\"subject\": \"training data\", \"predicate\": \"contains\", \"object\": \"disparities\"}, {\"subject\": \"gender\", \"predicate\": \"is\", \"object\": \"a factor\"}, {\"subject\": \"occupation\", \"predicate\": \"is\", \"object\": \"a factor\"}, {\"subject\": \"social norms\", \"predicate\": \"cause\", \"object\": \"disparities\"}]",

​    "summary": "The text describes how machine learning models can reflect and potentially amplify disparities found in the training data, especially when dealing with gender, occupation, and other social factors.",

​    "processed_at": "2026-01-21T07:27:33.052422"

  },

  {

​    "chunk_id": "926d51d0-77d9-45f8-8d69-6390c307a628",

​    "tags": "[\"disparate error rates\", \"population drift\", \"ethics\", \"causality\", \"feedback loops\"]",

​    "triplets": "[{\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"ethics\"}, {\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"feedback loops\"}, {\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"causality\"}]",

​    "summary": "The text describes the pitfalls of action in machine learning systems, focusing on disparate error rates for different groups, population drift, and the limitations of relying on predictions without understanding the underlying processes.",

​    "processed_at": "2026-01-21T07:28:28.725393"

  },

  {

​    "chunk_id": "73a46aa6-b4d9-47fc-aff2-4621040faed5",

​    "tags": "[\"feedback\", \"bias\", \"system\", \"predictions\", \"algorithm\"]",

​    "triplets": "[{\"subject\": \"feedback\", \"predicate\": \"uses\", \"object\": \"system\"}, {\"subject\": \"bias\", \"predicate\": \"causes\", \"object\": \"system\"}, {\"subject\": \"system\", \"predicate\": \"inherits_from\", \"object\": \"algorithm\"}, {\"subject\": \"predictions\", \"predicate\": \"enables\", \"object\": \"system\"}, {\"subject\": \"algorithm\", \"predicate\": \"part_of\", \"object\": \"system\"}]",

​    "summary": "The text describes the challenges and complexities of using feedback in systems, including the potential for bias and unintended consequences.",

​    "processed_at": "2026-01-21T07:29:36.389597"

  },

  {

​    "chunk_id": "ef7831c9-7a6c-40cf-bb85-f98df0ce30ef",

​    "tags": "[\"hiring committee\", \"GPA\", \"interview score\", \"job performance\", \"bias\"]",

​    "triplets": "[{\"subject\": \"hiring committee\", \"predicate\": \"uses\", \"object\": \"GPA\"}, {\"subject\": \"hiring committee\", \"predicate\": \"uses\", \"object\": \"interview score\"}, {\"subject\": \"hiring committee\", \"predicate\": \"uses\", \"object\": \"job performance\"}, {\"subject\": \"GPA\", \"predicate\": \"part_of\", \"object\": \"hiring process\"}, {\"subject\": \"interview score\", \"predicate\": \"part_of\", \"object\": \"hiring process\"}, {\"subject\": \"job performance\", \"predicate\": \"defined_as\", \"object\": \"hiring process\"}]",

​    "summary": "The text describes a hiring committee using GPA and interview score to predict job performance, and how this model can be biased against certain groups.",

​    "processed_at": "2026-01-21T07:30:47.867457"

  },

  {

​    "chunk_id": "4edab167-145b-45cb-b44c-8c322016e721",

​    "tags": "[\"group disparities\", \"ethical obligations\", \"decision making\", \"machine learning\", \"online search\", \"recommendation algorithms\"]",

​    "triplets": "[{\"subject\": \"group disparities\", \"predicate\": \"causes\", \"object\": \"ethical obligations\"}, {\"subject\": \"ethical obligations\", \"predicate\": \"enables\", \"object\": \"changes decisionmaking\"}, {\"subject\": \"decision making\", \"predicate\": \"uses\", \"object\": \"machine learning\"}, {\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"online search and recommendation algorithms\"}, {\"subject\": \"ethical obligations\", \"predicate\": \"requires\", \"object\": \"changing conditions\"}]",

​    "summary": "The book discusses the need for justice beyond fair decision-making, emphasizing the importance of changing conditions and systems that lead to unfair outcomes.",

​    "processed_at": "2026-01-21T07:31:33.390039"

  },

  {

​    "chunk_id": "6eb2c34c-9c91-4685-bd4c-b232b2d2581b",

​    "tags": "[\"machine learning\", \"fairness\", \"workplace dynamics\", \"automation\"]",

​    "triplets": "[{\"subject\": \"machine learning\", \"predicate\": \"uses\", \"object\": \"fair selection process\"}, {\"subject\": \"fair selection process\", \"predicate\": \"optimizes\", \"object\": \"job performance\"}, {\"subject\": \"workplace dynamics\", \"predicate\": \"enables\", \"object\": \"changes to the workplace environment\"}]",

​    "summary": "The text describes the importance of considering the broader workplace environment and its impact on job performance, especially when using machine learning for fair selection processes.",

​    "processed_at": "2026-01-21T07:32:29.307223"

  },

  {

​    "chunk_id": "3a3bd4e2-40b7-4674-9483-56ab2d1aa63a",

​    "tags": "[\"allocative harms\", \"representational harms\", \"gender stereotyping\", \"echo chambers\", \"algorithmic bias\", \"social media algorithms\"]",

​    "triplets": "[{\"subject\": \"allocative harms\", \"predicate\": \"causes\", \"object\": \"system withholds certain groups an opportunity or a resource\"}, {\"subject\": \"representational harms\", \"predicate\": \"reinforce\", \"object\": \"subordination of some groups along the lines of identity\"}, {\"subject\": \"gender stereotyping\", \"predicate\": \"reflect\", \"object\": \"prevalent gender composition and stereotypes about those occupations\"}, {\"subject\": \"echo chambers\", \"predicate\": \"exacerbate\", \"object\": \"political polarization\"}, {\"subject\": \"algorithmic bias\", \"predicate\": \"bias\", \"object\": \"content from the mainstream political right\"}, {\"subject\": \"social media algorithms\", \"predicate\": \"expose\", \"object\": \"content that conforms to user's prior beliefs\"}]",

​    "summary": "The text describes the harms of information systems, including the negative effects of search and recommendation algorithms, and the distinction between allocative and representational harms.",

​    "processed_at": "2026-01-21T07:33:38.029487"

  }

]
