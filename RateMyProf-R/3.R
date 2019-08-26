all_psy_uconn <-  all_psy %>%
  filter(school_name == "University of Connecticut")

all_psy_pennstate <-  all_psy %>%
  filter(school_name == "Pennsylvania State University")

all_psy_2 <- rbind(all_psy_uconn, all_psy_pennstate) #看一下两个测试学校的比较


ggplot(data = all_psy_2, mapping = aes(x = as.factor(course_level), fill = school_name)) +
  geom_bar(position = "dodge")


all_psy_top_schools_n <- all_psy %>% #最多数据的20所学校
  group_by(school_name) %>%
  summarise(n = n()) %>%
  ungroup() %>%
  arrange(-n)

all_psy_top_schools <- all_psy_top_schools_n[1:30,]$school_name

all_psy_top_schools_data <- all_psy %>%
  filter(school_name %in% all_psy_top_schools) %>%
  left_join(all_psy_top_schools_n)

all_psy_top_schools_data %>%
  ggplot(mapping = aes(x = course_level, fill = reorder(school_name, -n))) +
  geom_bar(position = "dodge")

all_psy_top_schools_data %>%
  ggplot(mapping = aes(x = as.numeric(student_difficult), y = as.numeric(student_star), color = reorder(school_name, -n))) +
  geom_smooth(method = "lm")

all_psy_top_schools_data %>%
  filter(course_level >=1, course_level <=4) %>%
  ggplot(mapping = aes(x = as.numeric(student_difficult), y = as.numeric(student_star), color = as.factor(course_level))) +
  geom_smooth(method = "lm") +
  facet_wrap(.~reorder(school_name, -n))


# if simply use course_level, different schools are not equally presented in different levels. The result is highly likely to be misunderstood.
# try to use 25%/50%/75%:
all_psy_top_schools_q <- all_psy_top_schools_data %>%
  group_by(school_name) %>%
  summarise(q.10 = quantile(course_level, 0.10),
            q.25 = quantile(course_level, 0.25),
            q.50 = quantile(course_level, 0.50),
            q.75 = quantile(course_level, 0.75),
            q.90 = quantile(course_level, 0.90),
            n = n()
            ) %>%
  ungroup() %>%
  arrange(-n)

all_psy_top_schools_data %>%
  ggplot(aes(x = reorder(school_name, -n), y = course_level)) +
  geom_violin() +
  geom_boxplot(width = 0.5) +
  coord_flip()
# 上面那个还是不太行 分不开课号

# 看一下分学校的话之前那个标签的规律还在不在
all_tags_level_school_summary <- all_tags_person %>%
  gather(key = "key", value = "value", 28:47) %>%
  filter(value != 0) %>%
  filter(course_level >=1, course_level <=4) %>%
  filter(school_name %in% all_psy_top_schools) %>%
  mutate(post_year = year(post_date_standard)) %>%
  group_by(school_name, course_level, key) %>%
  summarize(n = sum(value)) %>%
  group_by(school_name) %>%
  mutate(school_n = sum(n)) %>%
  group_by(school_name, course_level) %>%
  mutate(school_level_n = sum(n)) %>%
  group_by(course_level) %>%
  mutate(proportion = n / school_level_n) %>%
  ungroup()

ggplot(data = filter(all_tags_level_school_summary, course_level <= 4 & course_level >= 1),
       mapping = aes(x = key, y = proportion, fill = as.factor(course_level))) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  facet_wrap(.~reorder(school_name, -school_n))


# research plan
## 根据图选出阈值
qplot(all_psy_schools$n)

big_schools <- filter(all_psy_schools, n > 200)

all_psy_big_schools_data <- all_psy %>%
  right_join(big_schools)

big_schools_levels <- all_psy_big_schools_data %>%
  group_by(school_name, course_level) %>%
  summarize(n_level = n()) %>%
  ungroup() %>%
  spread(key = course_level, value = n_level)

# 选出university
big_universities <- big_schools %>%
  filter(str_detect(school_name, "University"),
         !str_detect(school_name, "Graduate"))

all_psy_big_universities_data <- all_psy %>%
  right_join(big_universities)

big_universities_levels <- all_psy_big_universities_data %>%
  group_by(school_name, course_level) %>%
  summarize(n_level = n()) %>%
  ungroup() %>%
  spread(key = course_level, value = n_level)

# 这些university里足够大的本科level 要求0:5 不能超过4个
big_u_big_l <- big_universities_levels %>%
  select(`school_name`:`5`)
big_u_big_l$state <- 0  #是否可用的状态
big_u_big_l$levels <- list(c(-1))

for(i in 1:nrow(big_u_big_l)){
  maxt <- max(as.numeric(big_u_big_l[i,]), na.rm = TRUE)
  for(j in 2:ncol(big_u_big_l)){
    if(!is.na(big_u_big_l[i,j]) & !is.na(maxt)){
      if(big_u_big_l[i,j] < 0.1*maxt){
        big_u_big_l[i,j] <- NA
      }
    }  
  }
  sumt <- sum(!is.na(as.numeric(big_u_big_l[i,])))
  levelt <- -1:5
  big_u_big_l$levels[i] <- list(levelt[!is.na(as.numeric(big_u_big_l[i,]))] )
  if(sumt <=4 & sumt >= 2){ big_u_big_l[i,]$state <- sumt} else{big_u_big_l[i,]$state <- 0}
}

big_u_big_l <- big_u_big_l %>%
  gather(key = "course_level", value = "n_level", `0`:`5`, na.rm = TRUE) 

#将这些level标准化
big_u_big_l <- big_u_big_l %>%
  filter(state != 0) %>%
  mutate(course_level_s = map2_dbl(
    .x = course_level,
    .y = levels,
    .f = function(x,y) return(which(y == x))
  )
  ) %>%
  mutate(course_level_s2 = map2_dbl(
    .x = course_level_s,
    .y = state,
    .f = function(x,y) return(scale(1:y)[,1][x])
  ))

# 将挑选出来的大学校大课与全体数据合并（大约剩下1/3）
big_u_big_l$course_level <- as.numeric(big_u_big_l$course_level)
big_u_big_l_data <- all_psy %>%
  right_join(big_u_big_l, by = c("school_name", "course_level"))

# 与标签数据合并
big_u_big_l_tags <- all_tags_person %>%
  right_join(big_u_big_l, by = c("school_name", "course_level"))

all_tags_category

classes <- c(4, 7, 12)
personality <- c(5, 6, 11, 13)
workload <- c(15, 16, 17, 18, 20)
grading <- c(2, 8, 10)

# all_tags_category_2
# 
# classes <- c(2, 7, 11)
# personality <- c(5, 6, 11, 13)
# workload <- c(15, 16, 17, 18, 20)
# grading <- c(2, 8, 10)

big_u_big_l_tags$classes <- 0
big_u_big_l_tags$personality <- 0
big_u_big_l_tags$workload <- 0
big_u_big_l_tags$grading <- 0

tag_sum <- function(df, v, i){
  sumt <- 0
  tagt <- tolower(all_tags_category[v])
  for (j in 1:ncol(df)) {
    if(  tolower(names(df)[j]) %in% tagt){
      sumt <- sumt + df[i,j]
    }
  }
  return(sumt)
}

for(i in 1:nrow(big_u_big_l_tags)){
  big_u_big_l_tags$classes[i] <- tag_sum(big_u_big_l_tags, classes, i)
  big_u_big_l_tags$personality[i] <- tag_sum(big_u_big_l_tags, personality, i)
  big_u_big_l_tags$workload[i] <- tag_sum(big_u_big_l_tags, workload, i)
  big_u_big_l_tags$grading[i] <- tag_sum(big_u_big_l_tags, grading, i)
  sumt <- sum(big_u_big_l_tags$classes[i], big_u_big_l_tags$personality[i], big_u_big_l_tags$workload[i], big_u_big_l_tags$grading[i], na.rm = TRUE)
  if(sumt != 0) {
    big_u_big_l_tags$classes[i] <- big_u_big_l_tags$classes[i]/sumt
    big_u_big_l_tags$personality[i] <- big_u_big_l_tags$personality[i]/sumt
    big_u_big_l_tags$workload[i] <- big_u_big_l_tags$workload[i]/sumt
    big_u_big_l_tags$grading[i] <- big_u_big_l_tags$grading[i]/sumt
  }
}



# 混合线性模型 检查是不是有年级趋势
library(lme4)
big_u_big_l_tags$school_name <- as.factor(big_u_big_l_tags$school_name)
big_u_big_l_tags$professor_name <- as.factor(big_u_big_l_tags$professor_name)
lm1 <-
with(big_u_big_l_tags,
     lmerTest::lmer(
       grading ~ course_level_s2 + (course_level_s2 | school_name / professor_name)#,
       #REML = FALSE
     ))

lm2 <-
  with(big_u_big_l_tags,
       lmerTest::lmer(
         classes ~ course_level_s2 + (course_level_s2 | professor_name)#,
         #REML = FALSE
       ))

#class 猜测少 显著少
#personality 猜测少 显著多
#grading 猜测多 显著多
#workload 猜测多 显著少
#数量级都在0.01附近 但是截距也就0.3