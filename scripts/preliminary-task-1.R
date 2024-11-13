require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# extract headings function
heading_fn <- function(.html){
  out = NULL
  for (i in 1:6){
    tag = paste("h", i, sep = "")
    content = read_html(.html) %>% 
      html_elements(tag) %>% 
      html_text2() %>% 
      str_c(collapse = ' ') %>% 
      rm_url() %>%
      rm_email() %>%
      str_remove_all('\'') %>%
      str_replace_all(paste(c('\n', 
                              '[[:punct:]]', 
                              'nbsp', 
                              '[[:digit:]]', 
                              '[[:symbol:]]'),
                            collapse = '|'), ' ') %>%
      str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
      toupper() %>%
      str_replace_all("\\s+", " ")
    out = paste(out, content) %>% 
      trimws()
  }
  out = trimws(out)
  return(out)
}

#loading claims data from example
load('../data/claims-clean-example.RData')

#creating headings column
claims_clean = claims_clean %>% 
  mutate(heading_clean = text_tmp)

for (i in 1:nrow(claims_clean)){
  claims_clean$heading_clean[i] = heading_fn(claims_clean$heading_clean[i])
}

save(claims_clean, file = "../data/claims-clean-task-1.RData")
