import React, { useEffect, useState } from 'react';
import Axios from 'axios';
import Button from '@material-ui/core/Button';
import { CircularProgress, Snackbar, Typography } from '@material-ui/core';
import Alert from '@material-ui/lab/Alert';
import { withStyles } from '@material-ui/core/styles';
import MaterialTable from 'material-table';
import SearchIcon from '@material-ui/icons/Search';

const styles = {
	predictButton: {
		width: '12rem',
	},
};

const getColumns = (keys, categoricalValues) =>
	keys
		.map(key =>
			key !== 'tableData'
				? {
						title: key,
						field: key,
						lookup: Object.keys(categoricalValues).includes(key)
							? categoricalValues[key].reduce((res, x) => {
									res[x] = x;
									return res;
							  }, {})
							: undefined,
						filterCellStyle: Object.keys(categoricalValues).includes(key)
							? undefined
							: { paddingTop: '32px' },
				  }
				: undefined,
		)
		.filter(val => val !== undefined);

const getDatasetDetails = id => Axios.get(`/api/datasets/${id}`);
const getDatasetSamples = id => Axios.get(`/api/datasets/${id}/retrieve_samples/`);

const Main = ({ classes, datasetId }) => {
	const [dataset, setDataset] = useState(undefined);
	const [error, setError] = useState(undefined);
	const [predicting, setPredicting] = useState(false);
	const [prediction, setPrediction] = useState('');
	const [allSamples, setSamples] = useState([]);
	const [categoricalValues, setCategoricalValues] = useState({});

	useEffect(() => {
		Axios.all([getDatasetDetails(datasetId), getDatasetSamples(datasetId)])
			.then(
				Axios.spread((details, samples) => {
					setDataset(details.data);
					setSamples(samples.data.samples);
					setCategoricalValues(samples.data.categorical_values);
				}),
			)
			.catch(_ =>
				setError(
					'Something went wrong while fetching the dataset. Please try another dataset or come back later.',
				),
			);

		return () => {
			setPrediction('');
			setSamples([]);
			setError(undefined);
			setDataset(undefined);
		};
	}, [datasetId]);

	const predictSample = (sampleId = -1) => {
		if (predicting) return;
		setPredicting(true);

		Axios.get(`/api/datasets/${datasetId}/predict_sample/`, { params: { sampleId } })
			.then(res => {
				setPrediction(res.data);
				setPredicting(false);
			})
			.catch(err => {
				setError(
					'Something went wrong while predicting the sample. Please try another sample or come back later.',
				);
				setPredicting(false);
			});
	};

	return (
		<React.Fragment>
			{allSamples.length > 0 ? (
				<React.Fragment>
					<MaterialTable
						title={dataset.title}
						data={allSamples}
						options={{
							pageSize: 10,
							pageSizeOptions: [5, 10, 20, 100],
							maxBodyHeight: '50vh',
							filtering: true,
						}}
						columns={getColumns(Object.keys(allSamples[0]), categoricalValues)}
						actions={[
							{
								icon: () => <SearchIcon />,
								tooltip: 'Predict sample',
								onClick: (event, rowData) => predictSample(rowData.tableData.id),
							},
						]}
					/>
					<Button
						variant="contained"
						color="primary"
						onClick={() => predictSample()}
						className={classes.predictButton}
					>
						{predicting ? (
							<CircularProgress size={26} color="white" />
						) : (
							<Typography>Predict a sample</Typography>
						)}
					</Button>
					{prediction !== '' && <h2>{prediction}</h2>}
				</React.Fragment>
			) : (
				<CircularProgress />
			)}
			<Snackbar open={error != null} autoHideDuration={6000} onClose={() => setError(undefined)}>
				<Alert onClose={() => setError(undefined)} severity="error">
					{error}
				</Alert>
			</Snackbar>
		</React.Fragment>
	);
};
export default withStyles(styles)(Main);
