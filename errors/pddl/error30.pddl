(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b2)
		(clear b2)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b1 t0)
		(in-tower b1)
		(on b8 b1)
		(in-tower b8)
		(on b3 b8)
		(in-tower b3)
		(red b0)
		(blue b1)
		(red b4)
		(yellow b5)
		(blue b6)
		(red b8)
		(yellow b9)
		(green b5)
		(blue b3)
		(green b7)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b4 b3)) (not (on b0 b3)))))
)