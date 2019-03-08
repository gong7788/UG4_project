(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(clear b9)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(green b0)
		(blue b0)
		(green b1)
		(blue b1)
		(green b2)
		(yellow b2)
		(green b3)
		(blue b3)
		(green b4)
		(blue b4)
		(green b5)
		(yellow b5)
		(green b6)
		(yellow b6)
		(red b7)
		(green b8)
		(blue b8)
		(green b9)
		(yellow b9)
		(green b7)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (not (on b5 b9))))
)