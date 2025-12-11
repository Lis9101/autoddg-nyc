11/27 Han Yang

我关于baseline的更新：
1. 我更改了baseline_autoddg.py，我加了header+sample(在baseline...json里的HandS列)这种描述方法，
它是通过给llm 列名和样例行，让llm直接生成一个短描述，这可以用来和autoddg生成的描述做对比。我也把SAMPLE, CONTENT, SEMANTIC这些原本的中间结果都存到输出json了，因为它们是后续evaluation中faithfulness的必要信息。
2. (重要) 我在baseline_autoddg.py的输出列里加了ufd_nyc和sfd_nyc用于接收autoddg-nyc的描述，但目前它和ufd和sfd一样，只是dummy列。
当开始做nyc特化prompt (week3任务)时，请在baseline里添加特化逻辑，然后把特化后的描述赋值给ufd_nyc和sfd_nyc，否则evaluation的pipeline无法正确生成相关评分。
3. 我还在baseline_autoddg.py的主函数run_baseline_autoddg里面加了个test_mode参数，设为False是原来的行为，设为True则只生成指定的50个数据集的描述json，方便快速测试。
4. 另外，我更改了baseline/semantic_autoddg.py。我从每个列都调用一次llm做语义概述，改成了所有列一起给llm，总共调用一次。这显著增加了效率，但如果数据集列数过多，可能降低llm表现。
5. 最后，文件结构做了点小改动，数据集的csv都在data/csv_files里，outputs存有所有的主要结果，比如baseline的输出文件，评估的输出文件和metadata_registry.json。


背景（不重要）：
论文总共提到了三种评估metrics:
ndcg：给定一个查询和ground truth，通过bm25 （Splade没用，因为太慢）按照每个数据集描述与查询的匹配度对数据集排序，排序与ground truth（真实排序）越接近，说明该生成数据集描述的方法越好
reference_free：无参考评估，即不需要除描述外的其它信息进行评估。通过LLM直接根据数据集描述评估completeness, conciseness, readability，通过给SAMPLE, CONTENT, SEMANTIC和描述信息来评估faithfulness(是否有幻觉)，所有分数都是0-10,最终结果归一化成0-1。
similarity：评估不同方法生成的描述与人类描述（原始描述）的相似性。指标有rouge等，这个评估没什么意义。


关于评估的重要说明：评估所需的输入文件就是baseline_autoddg_description.jsonl和metadata_registry.json，
text_eval.py (功能见下方文件说明)其实不需要baseline...json的全部2000个数据集描述，200个差不多就够了，只要有代表性，当然2000个也行。
ndcg_eval.py 则特殊一些，它要且仅要50个指定的数据集，所以你要么把baseline跑完让它从2000个里自动找，要么把baseline_autoddg.py里的test_mode打开，
指定生成这50个数据集的描述（你可以之后再关掉下载剩余的）。

所以如果你在测试不同的nyc特化prompt，你就让test_mode开着，改完一次，跑这50个数据集的描述，然后跑evaluator.py看表现，如此循环。
当你有一个最终特化版本后，再跑2000个，reference_free和similarity的评分会更精确一点。
如果你是第一次跑，ntlk可能会报错，因为有些nltk有些东西要一次性下载下。应该是下面这几个
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


快速使用：
python evaluator.py，生成的结果在outputs里。eval_radar.png和论文里的雷达图类似，eval_results.json是文本结果。


文件说明：
text_eval.py -> 实现reference_free和similarity的评估，结果文件是text_eval_results.jsonl
ndcg_eval.py -> 实现ndcg的评估，结果文件是ndcg_eval_results.json
python evaluator.py -> 调用text_eval.py和ndcg_eval.py，一键生成结果到outputs folder里

queries.txt -> 有代表性的20个查询
get_ndcg_ground_truth.py -> 让llm自动评判数据集对给定查询是否有相关性（0，1，2代表不相关，部分相关和相关），即生成ndcg的ground truth。
relevance_matrix.csv -> get_ndcg_ground_truth.py的输出文件，包含了50个有代表性的数据集和20个查询，这个用来做ground truth，用于后续NDCG打分。


