import pytest
import src

class TestCovidDataLoader:
    def setup_method(self):
        self.covid_data_loader = src.features.CovidDataLoader(level=1)

    def test_transform(self):
        df = self.covid_data_loader.transform(None)
        assert len(df) > 0


class TestTargetCreator:
    def setup_method(self):
        covid_data_loader = src.features.CovidDataLoader(level=1)
        target_creator = src.features.TargetCreator(shift_day=7)
        df = covid_data_loader.transform(None)
        df = target_creator.transform(df)
        self.df = df

    def test_transform(self):
        df = self.df
        target = df.loc[(df['location_key']=='ZA_WC') & (df['date']=='2022-05-01'), "target"].values[0]
        new_infection_after_7_days = df.loc[(df['location_key']=='ZA_WC') & (df['date']=='2022-05-08'), "new_infection_rate"].values[0]
        assert target == new_infection_after_7_days