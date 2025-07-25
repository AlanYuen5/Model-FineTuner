# import pandas as pd
# import json
# import argparse
# import os
# import openai
from logger import setup_logger
# import time
from gpt_fine_tuner import GPTFineTuner
from excel_to_jsonl import convert_excel_to_jsonl

# 设置日志
logger = setup_logger(__name__)


# def main():
#     logger.info("请输入Excel文件路径或拖拽文件到此处:")
#     excel_file_path = input()
#     excel_file_path = excel_file_path.strip("&").strip().strip("'")

#     logger.info(f"用户输入的文件路径: {excel_file_path}")

#     # 验证文件是否存在
#     if not os.path.exists(excel_file_path):
#         logger.error(f"文件不存在: {excel_file_path}")
#         logger.info("错误: 指定的文件不存在，请检查文件路径")
#         return

#     logger.info("开始转换Excel文件到JSONL格式")
#     convert_excel_to_jsonl(excel_file_path=excel_file_path, output_file_path="./data/file-first_check_data.jsonl")

#     fine_tuner = GPTFineTuner(
#         api_key="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"
#     )

#     # 上传训练和验证文件
#     training_file_id = fine_tuner.upload_file(
#         file_path="./data/file-first_check_training_data.jsonl"
#     )
#     validation_file_id = fine_tuner.upload_file(
#         file_path="./data/file-first_check_validation_data.jsonl"
#     )

#     # 目前的文件id对应着data文件夹里的first_check_training/validation_data.jsonl，无需再上传
#     training_file_id = "file-V3L9XcMF7hWiX4HLWiYRvo"
#     validation_file_id = "file-7TsvCrn8nJqryKqT76zLkr"


#     # 创建微调任务
#     job_id = fine_tuner.create_fine_tune_job(
#         training_file_id=training_file_id, validation_file_id=validation_file_id,
#         model="gpt-4.1-mini-2025-04-14"
#     )

#     # 监控微调任务，如需取消微调任务输入CTRL+C
#     # fine_tuner.monitor_training(job_id=job_id)

#     # 如果要在后端测试模型，取消注释以下代码，把model_id替换至程序微调完成后输出的model_id
#     test_message = """
#     深匹要求：
#     我们需要有过3年以上无人机销售经验的人。加分项是她需要有长续航的商用无人机的销售经验。
#     客户信息：
#     中文简历简历编号：e7d4952d988bCc90647344994|最后一次登录时间：2025/07/02查看大图今天活跃王**在职，看看新机会女35岁太原本科工作13年13k 大区销售经理北京梦之墨科技有限公司************立即沟通推荐职位查看联系方式收藏转发TA当前在线回复快，现在开始聊聊吧[表情]求职意向教育产品开发太原10-15k×13薪学校教育培训服务学术/科研查看全部 (2)工作经历北京梦之墨科技有限公司（2018.08 - 至今, 6年11个月）电子/半导体/集成电路100-499人大区销售经理薪　　资：13k 职位类别：区域销售经理/主管职责业绩：1、依据企业产品，挖掘市场应用场景及客户群体；
#     2、负责西北地区、山西省、山东省的高校客户及渠道挖掘、洽谈及合作；
#     3、寻找各区域各领域核心用户并促成合作，帮助渠道商建立标杆、样板客户；
#     4、组织各区域不同领域针对性潜在客户内部活动，以点带面快速渗透；
#     5、寻找国赛、省赛资源，作为其中一个模块嵌入到整体生意中；
#     4、培训、协助代理商进行产品销售、招投标等内容。
#     工作业绩：部分成交客户名单
#     西北工业大学、西安电子科技大学、西安交通大学、武警工程大学、兰州交通大学、兰州理工大学、太原工业学院、中北大学、太原学院、晋中职业技术学院、中国石油大学、青岛大学、山东大学等
#     北京韦加无人机科技股份有限公司（2016.07 - 2018.07, 2年）航空/航天设备100-499人区域总监所在部门：教育事业部职责业绩：1、依据企业项目，发展部分合作伙伴
#     2、负责山东省、山西省、陕西省中高职及部分本科院校校企合作市场的开发、洽谈，促成合作；
#     3、负责驻校代表人员招聘、任务部署、考核等工作，确保老客户维护及新客户开发；
#     4、带领团队完成合作院校无人机专业招生工作；
#     5、负责区域内省级大赛等营销活动策划及洽谈；
#     6、在合作院校进行筛选，进行短期培训考证合作；
#     7、进行招投标工作、学费收取，协调项目实施并保证项目回款。
#     工作业绩：
#     1、山东省专业共建合作 3所高职，17年招生人数为 121人，18年招生人数 220人（单招报名），年实训设备
#     采购中标金额 313万元；目前学校已按照韦加参数申请 2019年预算 450万。
#     2、陕西省专业共建合作 1所技师学院，16年招生人数 39人，17年招生人数为 43人，18年实训设备采购中标
#     金额为 197.06万元；1所高职 18年完成 200万预算实训室建设。
#     3、山西省合作 1所高职，18年完成 180万预算实训室建设。
#     4、2016 年签约 4所院校、2017年完成 400万业绩任务、2018年至今完成 240万业绩任务。中兴通讯股份有限公司（2015.01 - 2016.07, 1年6个月）通信设备10000人以上省级销售总监职责业绩：1、负责山西省 31所本科、48所高职院校针对通信、物联网、网络、电子信息、计算机、云计算/大数据方向、移动互联开发等信息类相关专业深度校企合作，其中包含专业实验室建设、师资团队培养、行业创新基地建设、校企专业共建；
#     2、教育部与中兴通讯面向全国高校 ICT行业/产教融合创新基地项目推广；
#     3、目标任务分解，团队配合，保证目标任务完成；
#     4、邀请陪同校级领导前往总部调研、深入洽谈合作项目；
#     5、意向达成项目进行招投标工作、协调项目实施并保证项目回款。
#     工作业绩：
#     1、完成山西交通职业技术学院 197万智慧交通实训设备采购。
#     2、与大同煤炭职业技术学院共同申报物联网及网络工程专业，共同培养学生，合作期为 10年，首届招生 51人。
#     3、与太原科技大学共同申报 ICT项目成功，完成签约，拟采购 500万实训设备。深圳国泰安教育技术股份有限公司山西办事处（2013.05 - 2014.12, 1年7个月）其他教育培训高级销售经理职责业绩：1、山西省 31所本科市场实训设备项目挖掘并进行 A/B/C类客户分类；
#     2、挖掘、引导客户需求，针对客户需求提供定制化整体解决方案；2 / 3
#     3、意向达成项目进行招投标工作、协调项目实施并保证项目回款；
#     4、定期组织客户参加师资培训班、大赛及校企合作交流会。
#     工作业绩：
#     1、完成太原工业学院、晋中学院、山西大学商务学院等累计 500万实训设备采购。
#     2、掌握山西省大部分高校二级教学及公共部门客户资源，部分校级领导资源。西安中盛凯新心理健康管理企业集团（2011.08 - 2013.04, 1年8个月）医疗器械渠道经理职责业绩：1、负责部署江苏省各地市健康管理中心项目宣传培训讲座；
#     2、江苏省每期招商会的筹办及代理人员的招募、洽谈、培训；
#     3、协助成立江苏省办事处。
#     工作业绩：
#     1、确定扬州、无锡、常州三家合作伙伴。
#     2、完成南京市两所三甲医院合作项目。
#     3、江苏省办事处正式成立。
#     目前薪金 Current salary：月发 1.4w*12+奖金+提成教育经历四川师范大学电影电视学院(现四川电影电视学院) · 播音与主持艺术 · 本科2008.09 - 2012.06统招本科在校时间与西安中盛凯新心理健康管理企业集团工作的时间有重叠询问TA语言能力普通话自我评价人选核心优势及面试评语 Core Advantages & General Comments:
#     1、本科就读播音与主持艺术专业，并攻读了法学学位，形象气质佳、逻辑思维清晰、语言沟通表达能力好、善于引导；
#     2、毕业后开始从事市场销售工作，经过 10年市场销售历练，熟练掌握市场整体规划、客户资源分类、用户痛点挖掘、时间分配、优势谈判、专业学习、团队合作、项目进度把控等能力。
#     3、经过对山西省、山东省、陕西省、甘肃省等高校教育市场耕耘，有一定的有效客户资源。
#     4、倾向电工电子、人工智能类、信息类、智能制造类专业方向工作。附件简历/作品TA上传了附件简历/作品，点击开聊向TA索要向TA索要本次搜索匹配到的简历不够精准？点击这里进行反馈简历不匹配声明：该人选信息仅供公司招聘使用，严禁以招聘以外的任何目的使用人选信息或利用猎聘平台及人选信息从事任何违法违规活动。否则，猎聘有权单方决定采取包括但不限于删除发布内容，限制、暂停使用，终止合作永久封禁账户等措施。简历备注共0条添加备注简历洞察一秒洞察名企背景、在职学历、工作空档期…亮点：2注意点：1去查看简历信息求职意向工作经历教育经历
#     """

#     system_prompt = """
#     我将给到你一个候选人的个人信息，岗位要求和加分项，请根据你对于公司，行业，岗位的理解，一步一步分析判断这个候选人是否符合岗位要求，你的判断请基于以下几个准则： 1 我提供的候选人个人信息里包含了这个人的个人介绍、工作经历、所在公司以及所在公司的介绍等，请综合考虑他所在公司的信息和个人信息进行判断。 2 如果你基于候选人的个人信息判断这个人是符合岗位要求的，所属行业以及在职的职位是符合要求的，并且能胜任这个职位，则回复"匹配"并解释可以胜任的原因；（可以胜任这个岗位的人，应该是和我提供的岗位要求里描述的需要的人的背景、负责的产品、所在的行业、曾经的履历、负责的市场或者研发环节是相似的，并且不是我们不想要的人。） 3 如果你的判断这个人是符合岗位要求的，并且他还符合加分项要求，那么这个人是非常胜任这个职位，则回复"非常匹配"并解释非常匹配的原因。 4 如果你基于候选人的个人信息无法判断这个人是否符合岗位要求，例如所属行业或者职位只有一部分符合要求，另外一部分不符合要求，或者信息缺失无法判断的，则回复"不确定"并解释原因（原因中需要包含这个人主要从事的行业，主要负责的产品或服务以及职位类型，主要负责的市场（国家或者地区）及规模或者主要负责的研发的具体环节）。 5 在候选人所在行业和职位是否符合我们的岗位要求判断时，请只从他的当前职位和公司以及上一段经历来判断。但是，他的上一段经历不符合行业或者职位要求是不影响他是否符合这个岗位要求的判断的。 6 如果当前的工作经历不符合要求，但是上一段工作经历符合要求，则不确定候选人是否能胜任这个岗位，则回复"不确定"并解释原因。 7 岗位要求中的关于候选人不能是什么样的人这样的要求指的是候选人最近一段时间的工作不能是这样的，之前从事过不符合要求的工作，现在从事的工作符合要求，则候选人是符合要求的。 5 原因和判断用"，"隔开。 具体输出举例如下： 例子1 输入的岗位要求是："岗位要求：我们需要的是地板或者建材行业的销售或者商务拓展人员，不是高管。" 当符合要求时的输出可能是："匹配，这个人最近履历是LVT地板的商务BD，并且是专员级别，不是高管，符合要求。" 当不符合要求时的输出可能是："不匹配，这个人是水泥厂的厂长，不是地板或者建材行业，所在公司是山东的水泥生产公司，主要是负责水泥生产的运营管理工作，曾经的履历也和地板销售不相关。" 当不确定是否符合要求时的输出可能是："不确定，这个人没有描述具体销售的产品，所在公司是零售超市，他是销售顾问，曾经的履历是广告营销，没有地板或者建材行业销售经验。" 例子2 输入的岗位要求是："岗位要求：我们需要的是做过外包人力资源服务的招聘顾问或者经理，并且主要负责外资企业的招聘工作，并且不要只做快消品公司招聘的人。加分项是有超过3年工作经验" 当符合要求时的输出可能是："非常匹配，这个人在外资企业担任招聘顾问，主要负责外包服务，并且有超过3年工作经验，满足加分项。" 当符合要求时的输出可能是："匹配，这个人所在公司锐仕方达是人力资源外包公司，他是猎头顾问，属于相关职务，曾经的履历有提到他主要负责的是思科、微软等公司的招聘，这些公司都是外资企业，并且不是快消品公司。" 当不符合要求时的输出可能是："不匹配，这个人是人力资源外包公司的财务，他主要负责代理记账和审计工作，没有做过外包人力资源服务。" 当不确定是否符合要求时的输出可能是："不确定，这个人目前是在互联网科技公司做招聘，上一段工作是在外资猎头公司做猎头顾问，当前工作不符合要求，上一段工作符合要求，所以不确定是否匹配。" --- 你需要记忆的重要概念： 1 海外工作经验：指的是这个人在非中国地区有过工作经验，在中国的外资公司工作不属于海外工作经验。 2 海外销售经验：指的是这个人负责人过非中国地区的销售，在中国的外资企业做销售不属于具有海外销售经验。 3 外包服务经验：指的是这个人在人力资源公司或者企业管理或者咨询或者事务所类型公司工作，为其他公司提供服务，这类公司也叫乙方公司。 4 超大型公司：指的是大型财团或者集团公司（那些在多个领域都有业务的公司或者主要做投资的公司，例如蒙牛是超大型公司，欣旺达不是超大型公司，上市公司不一定是超大型公司）。 5 行业判断：在进行候选人行业判断时，你主要根据公司的名字、候选人描述的工作内容和负责的产品来判断。 6 销售类岗位：一般商务、大客户、销售、客户关系都可以是销售类岗位。 7 主要负责某项业务：指的是候选人在最近5年的经历中相比于其他业务，更多时间或者经历投入在某类工作内容或者职责中。 8 不要只做过某项业务的人：指的是这个岗位不需要只做过某项任务，没有做过其他任务的候选人。 9 研发生产环节：在判断候选人是否匹配时，你需要注意候选人所负责的工作内容是否是和岗位所要求的工作环节对应，这些工作环节相互之间是不一样的：研发，工艺，生产管理，设备管理，结构设计，嵌入式硬件，嵌入式软件，测试，产品设计。 10 产品设计/产品管理：一般指的不是金融类产品的设计和管理，除非在岗位要求中专门讲了是金融产品。 --- 你的回答用中文，控制在100个字以内。无论是否匹配，解释的原因中都需要包含这个人主要从事的行业，主要负责的产品或服务以及职位类型，主要负责的市场（国家或者地区）及规模或者研发负责的具体工作环节。这样方便我去理解你判断的是否准确。对于不匹配的人必须说明是哪一个原因导致的不匹配，而匹配的人只需说明是哪些项目匹配，不需要提及加分项。
#     """

#     # old model 
#     model_id = "ft:gpt-4.1-mini-2025-04-14:anyhelper::BpY0RELu"
#     # new worse model
#     # model_id = "ft:gpt-4.1-mini-2025-04-14:anyhelper::BqWwlvgX"    

#     result = fine_tuner.ask_model(model_id=model_id, system_prompt=system_prompt, test_message=test_message, temperature=0.0)


import streamlit.web.bootstrap

if __name__ == "__main__":
    # import os
    # os.system("python -m streamlit run main_ui.py")
    streamlit.web.bootstrap.run('main_ui.py', False, [], {})