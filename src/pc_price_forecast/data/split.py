import pandas as pd
from sklearn.model_selection import train_test_split
from ..config import TEST_SIZE, VALID_SIZE, RANDOM_SEED

def split_train_valid_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # valid is a fraction of train_df
    valid_frac_of_train = VALID_SIZE
    train_df, valid_df = train_test_split(train_df, test_size=valid_frac_of_train, random_state=RANDOM_SEED)

    return train_df, valid_df, test_df
