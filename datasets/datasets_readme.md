# 1、新疆光伏数据集(`XJ_Photovoltaic.csv`)

原始数据集的核心字段包括：
- 日期列：`date`，以 YYYY-MM-DD hh:mm:ss 形式呈现年月日、时分秒信息
- 组件温度：`components temperature(℃)`   
- 气温：`temperature(℃)`
- 气压：`air pressure(hPa)`
- 湿度：`humidity(%)`
- 总辐射：`total radiation(W/m2)`
- 直射幅度：`direct radiation(W/m2)`
- 散射幅度：`diffuse radiation(W/m2)`
- 实际发电功率：`load(mw)`

时间间隔为**15min**，整个数据集的时间跨度长达**一年**，从`2019-01-01 00:00:00`到`2019-12-31 23:45:00`，包含**35041条数据**。在实验中我们把**实际发电功率`load`**作为预测目标`target`。


# 2、澳大利亚电力负荷与价格预测数据(`Australia_Load&Price.csv`)

原始数据集的核心字段包括：
- 日期列：`date`，以 YYYY/M/D h:m 形式呈现年月日、时分信息 
- 干球温度：`dry bulb temperature(℃)`
- 露点温度：`dew point temperature(℃)`
- 湿球温度：`wet bulb temperature(℃)`
- 湿度：`humidity`
- 电价：`electrcity price`
- 电力负荷：`load`

时间间隔为**30min**，整个数据集的时间跨度长达**四年**，从`2006/1/1 0:00`到 `2010/12/31 23:30`，包含**87648条数据**。在实验中我们把**电力负荷`load`**作为预测目标`target`。


# 3、2016电工数学建模竞赛负荷预测数据集（`Mathematical_Modeling_Competition.csv`）

原始数据集的核心字段包括：
- 日期列：`date`，以 YYYYMMDD 形式呈现年月日信息
- 最高温度：`Max_temperature(℃)`
- 最低温度：`Min_temperature(℃)`
- 平均温度：`Average_temperature(℃)`
- 平均相对湿度：`Relative_humidity(average)`
- 降雨量：`Rainfall(mm)`
- 日需求负荷：`load`
时间间隔为**1天**，整个数据集的时间跨度长达**三年**，从`20120101`到`20150110` ，包含**1106条数据**。在实验中我们把**日需求负荷`load`**作为预测目标 `target`。

# 4、巴拿马国家电力负荷数据集（`Panama_CND.csv`）

数据集已经预处理完成。可直接用于时间序列预测任务，包含巴拿马国家三个主要城市的历史电力负荷数据及多维度影响因素，字段说明如下：

1. `datetime`：YYYY-MM-DD HH:MM:SS 格式的时间戳（每小时粒度）

2. `nat_demand`：国家电力负荷（需求值），源自国家电网运营商(CND)的日度后调度报告。

3. `T2M_toc`~`W2M_dav`（12列）：气象数据，源自 NASA Earthdata，按3个城市分组，每个城市包含4个气象指标，其中带`_toc`、`_san`、`_dav`后缀的分别表示巴拿马城（Tocumen ）、圣地亚哥市（Santiago）、大卫城（David），带`T2M_`、`QV2M_`、`TQL_`、`W2M_`前缀的分别指2米高度温度(℃)、2米高度相对湿度、液态降水量、2米高度风速(m/s)。
    
4. `Holiday_ID`~`school`：日历特征，具体而言，
    - `Holiday_ID`：节假日唯一标识码
    - `holiday`：信息源于"When on Earth?"网站，指示节假日标识，其中 1=节假日，0=普通日
    - `school`：信息源于巴拿马教育部，指示学期期间标识，其中 1=学期中，0=假期

时间间隔为**1h**，整个数据集时间跨度长度五年有余，从`2015-01-03 01:00:00`到`2020-06-27 00:00:00`，包含**48049条数据**。在实验中我们把**国家需求负荷**`nat_demand`作为预测目标 `target`。


